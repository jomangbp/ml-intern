"""
EGGROLL vs Backprop Comparison Experiment
==========================================

Fair head-to-head comparison of:
1. EGGROLL (gradient-free, arXiv:2511.16652) 
2. Standard Backprop (AdamW)

Controls (identical for both):
- Same model architecture (HybridSLM ~74M params)
- Same data (TinyStories, train_tokens.bin)
- Same optimizer steps: 5,900
- Same batch config: bs=3, accum=2, seq=1024
- Same hardware (RTX 3060 Laptop 6GB)
- Same eval schedule (every 1,000 steps)
- Same logging (Trackio)

Only variable: training method.

Usage:
    # Run both methods sequentially
    python scripts/egroll_comparison.py --method both

    # Run only EGGROLL
    python scripts/egroll_comparison.py --method egroll

    # Run only backprop reference
    python scripts/egroll_comparison.py --method backprop

Output:
    outputs/egroll-experiment/
    ├── egroll/          (EGGROLL results)
    └── backprop/        (Backprop reference results)

Trackio: project="hybrid-slm", run="egroll-vs-backprop"
"""

import os
import sys
import math
import time
import gc
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

import numpy as np

import trackio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, HybridSLMModel
from configs.model_config import HybridSLMConfig
from src.egroll import EGGROLLConfig, LowRankPerturbation


# ══════════════════════════════════════════════════════════════════
# Shared data utilities
# ══════════════════════════════════════════════════════════════════

class TokenizedDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = 1024, stride: int = 256):
        self.max_length = max_length
        self.stride = stride
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Not found: {data_path}")
        self.ids = np.memmap(data_path, dtype=np.uint32, mode="r")
        self.size = max(0, len(self.ids) - (max_length + 1))

    def __len__(self):
        return max(0, self.size // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length + 1
        window = self.ids[start:end].astype(np.int64)
        return {
            "input_ids": torch.from_numpy(window[:-1]),
            "labels": torch.from_numpy(window[1:]),
        }


class StreamingTokenDataset(IterableDataset):
    def __init__(self, data_path: str, max_length: int = 1024, stride: int = 256,
                 chunk_samples: int = 64_000, seed: int = 42):
        self.data_path = data_path
        self.max_length = max_length
        self.stride = stride
        self.chunk_samples = chunk_samples
        self.seed = seed
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Not found: {data_path}")
        ids_tmp = np.memmap(data_path, dtype=np.uint32, mode="r")
        self.num_tokens = len(ids_tmp)
        self.num_samples = max(0, (self.num_tokens - (max_length + 1)) // stride)
        del ids_tmp

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        ids = np.memmap(self.data_path, dtype=np.uint32, mode="r")
        num_samples = self.num_samples
        chunk_size = self.chunk_samples

        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            token_start = chunk_start * self.stride
            token_end = (chunk_end - 1) * self.stride + self.max_length + 1
            big_chunk = ids[token_start:token_end].astype(np.int64)

            samples = []
            for i in range(chunk_start, chunk_end):
                local_start = (i - chunk_start) * self.stride
                local_end = local_start + self.max_length + 1
                window = big_chunk[local_start:local_end]
                samples.append({
                    "input_ids": torch.from_numpy(window[:-1].copy()),
                    "labels": torch.from_numpy(window[1:].copy()),
                })

            rng.shuffle(samples)
            for item in samples:
                yield item
            del big_chunk, samples


def make_dataloaders(batch_size=3, max_length=1024):
    """Create train/val dataloaders using TinyStories data."""
    train_path = "data/train_tokens.bin"
    val_path = "data/val_tokens.bin"

    train_ds = StreamingTokenDataset(train_path, max_length=max_length, chunk_samples=32_000)
    val_ds = TokenizedDataset(val_path, max_length=max_length)

    collate = lambda x: {
        'input_ids': torch.stack([i['input_ids'] for i in x]),
        'labels': torch.stack([i['labels'] for i in x]),
    }

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0,
                              pin_memory=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, val_loader, num_batches=50):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        del outputs, loss, input_ids, labels
        torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens
    return {'val_loss': avg_loss, 'val_ppl': math.exp(min(avg_loss, 20))}


# ══════════════════════════════════════════════════════════════════
# EGGROLL training
# ══════════════════════════════════════════════════════════════════

class EgRollExperiment:
    """
    EGGROLL training on the baseline HybridSLM.
    
    Uses low-rank perturbations (rank-2) with population-based evaluation.
    Each step: evaluate N perturbed models → z-score normalize rewards → weighted update.
    
    From arXiv:2511.16652:
    - Update: θ_{t+1} = θ_t + (α/N) Σ_i ε_i * f(θ_t + σ*ε_i)
    - Low-rank: ε = AB^T where A∈R^{m×r}, B∈R^{n×r}
    """

    def __init__(self, output_dir, population_size=64, rank=2,
                 noise_scale=0.001, learning_rate=5e-4, max_steps=5900):
        self.output_dir = output_dir
        self.population_size = population_size
        self.rank = rank
        self.noise_scale = noise_scale
        self.learning_rate = learning_rate
        self.max_steps = max_steps

        self.device = "cuda"
        self.model_config = HybridSLMConfig()
        self.model = create_model(self.model_config).to(self.device)
        self.model.eval()

        # Cache parameter structure
        self.param_names = []
        self.param_shapes = []
        self.param_refs = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.is_leaf:
                self.param_names.append(name)
                self.param_shapes.append(tuple(param.shape))
                self.param_refs.append(param)

        self.best_val_loss = float('inf')
        self.best_reward = float('-inf')

    def _fitness(self, model, batch):
        """Negative CE loss (higher = better)."""
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
            loss = outputs['loss']
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                return -20.0
            return -loss.item()

    def _step(self, batch):
        """Single EGGROLL optimization step."""
        # Snapshot parameters
        original = {n: p.data.clone() for n, p in zip(self.param_names, self.param_refs)}

        rewards = []
        perts_by_layer = [[] for _ in self.param_names]

        for _ in range(self.population_size):
            pert_state = {}
            for i, (name, shape) in enumerate(zip(self.param_names, self.param_shapes)):
                lr_p = LowRankPerturbation(shape, self.rank, self.device)
                eps = lr_p.sample()
                pert_state[name] = original[name] + self.noise_scale * eps
                perts_by_layer[i].append(eps)

            self.model.load_state_dict(pert_state, strict=False)
            rewards.append(self._fitness(self.model, batch))

        # Restore original
        full_state = self.model.state_dict()
        for name in original:
            full_state[name] = original[name]
        self.model.load_state_dict(full_state)

        # Z-score normalize rewards
        r = torch.tensor(rewards, device=self.device)
        r_mean, r_std = r.mean(), r.std()
        if r_std < 1e-8:
            r_std = torch.tensor(1.0, device=self.device)
        norm_r = (r - r_mean) / r_std

        # Weighted update
        with torch.no_grad():
            for i, pref in enumerate(self.param_refs):
                update = sum(norm_r[j] * perts_by_layer[i][j]
                             for j in range(len(rewards)))
                pref.data.add_(self.learning_rate * update / self.population_size)

        self.best_reward = max(self.best_reward, r_mean.item())
        return {
            'reward_mean': r_mean.item(),
            'reward_std': r_std.item(),
            'estimated_loss': -r_mean.item(),
        }

    def run(self):
        """Run EGGROLL training."""
        print("\n" + "=" * 60)
        print("EGGROLL TRAINING (gradient-free)")
        print("=" * 60)
        print(f"  Population: {self.population_size}")
        print(f"  Rank: {self.rank}")
        print(f"  Noise σ: {self.noise_scale}")
        print(f"  LR α: {self.learning_rate}")
        print(f"  Steps: {self.max_steps}")

        train_loader, val_loader = make_dataloaders(batch_size=3)
        train_iter = iter(train_loader)

        os.makedirs(self.output_dir, exist_ok=True)
        log_file = open(os.path.join(self.output_dir, "training_log.txt"), "w")

        start_time = time.time()
        total_tokens = 0

        for step in range(self.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            gpu_batch = {
                'input_ids': batch['input_ids'].cuda(non_blocking=True),
                'labels': batch['labels'].cuda(non_blocking=True),
            }

            metrics = self._step(gpu_batch)
            total_tokens += gpu_batch['input_ids'].numel() * self.population_size

            # Log every 10 steps
            if (step + 1) % 10 == 0:
                elapsed = time.time() - start_time
                tok_s = total_tokens / elapsed if elapsed > 0 else 0
                mem = torch.cuda.memory_allocated() / 1024**3

                line = (f"step={step+1:,} | est_loss={metrics['estimated_loss']:.4f} | "
                        f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f} | "
                        f"tok/s={tok_s:,.0f} | mem={mem:.2f}GB")
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

                trackio.log({
                    "egroll/estimated_loss": metrics['estimated_loss'],
                    "egroll/reward_mean": metrics['reward_mean'],
                    "egroll/reward_std": metrics['reward_std'],
                    "egroll/best_reward": self.best_reward,
                    "egroll/tokens_per_sec": tok_s,
                    "egroll/tokens_seen": total_tokens,
                    "egroll/gpu_memory_gb": mem,
                })

            # Eval every 1000 steps
            if (step + 1) % 1000 == 0:
                print(f"\n--- EGGROLL Eval at step {step+1} ---")
                ev = evaluate(self.model, val_loader, num_batches=50)
                print(f"  val_loss={ev['val_loss']:.4f} | val_ppl={ev['val_ppl']:.2f}")
                log_file.write(f"  val_loss={ev['val_loss']:.4f} | val_ppl={ev['val_ppl']:.2f}\n")
                log_file.flush()

                trackio.log({"egroll/val_loss": ev['val_loss'],
                             "egroll/val_ppl": ev['val_ppl']})

                if ev['val_loss'] < self.best_val_loss:
                    self.best_val_loss = ev['val_loss']
                    bp = os.path.join(self.output_dir, "best")
                    os.makedirs(bp, exist_ok=True)
                    torch.save(self.model.state_dict(), f"{bp}/pytorch_model.bin")
                    print(f"  ★ New best (val_loss={ev['val_loss']:.4f})")

                self.model.cuda()
                torch.cuda.empty_cache()

            # Periodic cleanup
            if step % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            del gpu_batch

        # Final
        fp = os.path.join(self.output_dir, "final")
        os.makedirs(fp, exist_ok=True)
        torch.save(self.model.state_dict(), f"{fp}/pytorch_model.bin")

        ev = evaluate(self.model, val_loader, num_batches=100)
        print(f"\nEGGROLL Final: val_loss={ev['val_loss']:.4f} | val_ppl={ev['val_ppl']:.2f}")
        log_file.write(f"\nFinal val_loss={ev['val_loss']:.4f} | val_ppl={ev['val_ppl']:.2f}\n")
        log_file.close()

        trackio.log({"egroll/final_val_loss": ev['val_loss'],
                     "egroll/final_val_ppl": ev['val_ppl']})

        # Save summary
        summary = {
            "method": "egroll",
            "steps": self.max_steps,
            "population": self.population_size,
            "rank": self.rank,
            "noise_scale": self.noise_scale,
            "lr": self.learning_rate,
            "final_val_loss": ev['val_loss'],
            "final_val_ppl": ev['val_ppl'],
            "best_val_loss": self.best_val_loss,
            "best_reward": self.best_reward,
            "training_time_min": (time.time() - start_time) / 60,
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        del self.model
        torch.cuda.empty_cache()
        gc.collect()

        return summary


# ══════════════════════════════════════════════════════════════════
# Backprop training (reference)
# ══════════════════════════════════════════════════════════════════

class BackpropExperiment:
    """
    Standard AdamW backprop training on the baseline HybridSLM.
    Identical config to the original baseline run (5,900 steps).
    """

    def __init__(self, output_dir, max_steps=5900):
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.device = "cuda"

        self.model_config = HybridSLMConfig()
        self.model = create_model(self.model_config).to(self.device)

        self.best_val_loss = float('inf')

    def run(self):
        print("\n" + "=" * 60)
        print("BACKPROP TRAINING (AdamW) — Reference")
        print("=" * 60)

        lr = 4e-4
        min_lr = 4e-5
        warmup = 2000
        accum_steps = 2
        batch_size = 3

        print(f"  LR: {lr} | Warmup: {warmup} | Min LR: {min_lr}")
        print(f"  Batch: {batch_size} × {accum_steps} accum")
        print(f"  Steps: {self.max_steps}")

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.1,
                          betas=(0.9, 0.95), eps=1e-8)
        scaler = GradScaler('cuda')

        train_loader, val_loader = make_dataloaders(batch_size=batch_size)
        train_iter = iter(train_loader)

        os.makedirs(self.output_dir, exist_ok=True)
        log_file = open(os.path.join(self.output_dir, "training_log.txt"), "w")

        start_time = time.time()
        total_tokens = 0

        for step in range(self.max_steps):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                input_ids = batch['input_ids'].cuda(non_blocking=True)
                labels = batch['labels'].cuda(non_blocking=True)

                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss'] / accum_steps

                scaler.scale(loss).backward()
                step_loss += loss.item() * accum_steps
                total_tokens += input_ids.numel()

                del outputs, loss, input_ids, labels

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Cosine LR
            s = step + 1
            if s < warmup:
                cur_lr = lr * s / warmup
            else:
                progress = (s - warmup) / (self.max_steps - warmup)
                cur_lr = min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            # Log every 10 steps
            if s % 10 == 0:
                elapsed = time.time() - start_time
                tok_s = total_tokens / elapsed if elapsed > 0 else 0
                mem = torch.cuda.memory_allocated() / 1024**3
                avg = step_loss / accum_steps

                line = (f"step={s:,} | loss={avg:.4f} | lr={cur_lr:.2e} | "
                        f"tok/s={tok_s:,.0f} | mem={mem:.2f}GB")
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

                trackio.log({
                    "backprop/loss": avg,
                    "backprop/learning_rate": cur_lr,
                    "backprop/tokens_per_sec": tok_s,
                    "backprop/tokens_seen": total_tokens,
                    "backprop/gpu_memory_gb": mem,
                })

            # Eval every 1000 steps
            if s % 1000 == 0:
                print(f"\n--- Backprop Eval at step {s} ---")
                ev = evaluate(self.model, val_loader, num_batches=50)
                print(f"  val_loss={ev['val_loss']:.4f} | val_ppl={ev['val_ppl']:.2f}")
                log_file.write(f"  val_loss={ev['val_loss']:.4f} | val_ppl={ev['val_ppl']:.2f}\n")
                log_file.flush()

                trackio.log({"backprop/val_loss": ev['val_loss'],
                             "backprop/val_ppl": ev['val_ppl']})

                if ev['val_loss'] < self.best_val_loss:
                    self.best_val_loss = ev['val_loss']
                    bp = os.path.join(self.output_dir, "best")
                    os.makedirs(bp, exist_ok=True)
                    torch.save(self.model.state_dict(), f"{bp}/pytorch_model.bin")
                    print(f"  ★ New best (val_loss={ev['val_loss']:.4f})")

                self.model.cuda()
                torch.cuda.empty_cache()

            # Cleanup
            if s % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # Final
        fp = os.path.join(self.output_dir, "final")
        os.makedirs(fp, exist_ok=True)
        torch.save(self.model.state_dict(), f"{fp}/pytorch_model.bin")

        ev = evaluate(self.model, val_loader, num_batches=100)
        print(f"\nBackprop Final: val_loss={ev['val_loss']:.4f} | val_ppl={ev['val_ppl']:.2f}")
        log_file.write(f"\nFinal val_loss={ev['val_loss']:.4f} | val_ppl={ev['val_ppl']:.2f}\n")
        log_file.close()

        trackio.log({"backprop/final_val_loss": ev['val_loss'],
                     "backprop/final_val_ppl": ev['val_ppl']})

        summary = {
            "method": "backprop",
            "steps": self.max_steps,
            "lr": lr,
            "warmup": warmup,
            "final_val_loss": ev['val_loss'],
            "final_val_ppl": ev['val_ppl'],
            "best_val_loss": self.best_val_loss,
            "training_time_min": (time.time() - start_time) / 60,
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        del self.model
        torch.cuda.empty_cache()
        gc.collect()

        return summary


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EGGROLL vs Backprop Comparison")
    parser.add_argument("--method", choices=["egroll", "backprop", "both"],
                        default="both", help="Which method(s) to run")
    parser.add_argument("--max-steps", type=int, default=5900)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--noise-scale", type=float, default=0.001)
    parser.add_argument("--egroll-lr", type=float, default=5e-4)
    parser.add_argument("--output-dir", type=str,
                        default="outputs/egroll-experiment")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EGGROLL vs BACKPROP COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"  Steps:      {args.max_steps}")
    print(f"  Population: {args.population}")
    print(f"  Rank:       {args.rank}")
    print(f"  Noise σ:    {args.noise_scale}")
    print(f"  EGGROLL LR: {args.egroll_lr}")
    print(f"  Method:     {args.method}")
    print(f"  Output:     {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Trackio init
    trackio.init(
        project="hybrid-slm",
        name="egroll-vs-backprop",
        config={
            "experiment": "egroll-vs-backprop",
            "model": "hybrid-slm-baseline (~74M params)",
            "max_steps": args.max_steps,
            "population_size": args.population,
            "rank": args.rank,
            "noise_scale": args.noise_scale,
            "egroll_lr": args.egroll_lr,
            "backprop_lr": 4e-4,
            "dataset": "TinyStories (25M tokens)",
            "seq_length": 1024,
            "batch_size": "3 × 2 accum = 6144 tokens/step",
        },
    )

    results = {}

    # ── Phase 1: EGGROLL ────────────────────────────────────────
    if args.method in ("egroll", "both"):
        eg_dir = os.path.join(args.output_dir, "egroll")
        eg = EgRollExperiment(
            output_dir=eg_dir,
            population_size=args.population,
            rank=args.rank,
            noise_scale=args.noise_scale,
            learning_rate=args.egroll_lr,
            max_steps=args.max_steps,
        )
        results['egroll'] = eg.run()

    # ── Phase 2: Backprop ───────────────────────────────────────
    if args.method in ("backprop", "both"):
        bp_dir = os.path.join(args.output_dir, "backprop")
        bp = BackpropExperiment(output_dir=bp_dir, max_steps=args.max_steps)
        results['backprop'] = bp.run()

    # ── Final comparison ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)

    if 'egroll' in results and 'backprop' in results:
        eg = results['egroll']
        bp = results['backprop']

        print(f"\n{'Metric':<25} {'EGGROLL':>12} {'Backprop':>12} {'Winner':>10}")
        print("-" * 62)
        print(f"{'Best val_loss':<25} {eg['best_val_loss']:>12.4f} {bp['best_val_loss']:>12.4f} "
              f"{'← Backprop' if bp['best_val_loss'] < eg['best_val_loss'] else '← EGGROLL':>10}")
        print(f"{'Final val_loss':<25} {eg['final_val_loss']:>12.4f} {bp['final_val_loss']:>12.4f} "
              f"{'← Backprop' if bp['final_val_loss'] < eg['final_val_loss'] else '← EGGROLL':>10}")
        print(f"{'Final val_ppl':<25} {eg['final_val_ppl']:>12.2f} {bp['final_val_ppl']:>12.2f} "
              f"{'← Backprop' if bp['final_val_ppl'] < eg['final_val_ppl'] else '← EGGROLL':>10}")
        print(f"{'Training time (min)':<25} {eg['training_time_min']:>12.1f} {bp['training_time_min']:>12.1f}")
        print(f"{'Method':<25} {'ES (grad-free)':>12} {'AdamW':>12}")

        # Save comparison
        comparison = {
            "egroll": eg,
            "backprop": bp,
            "winner": "backprop" if bp['final_val_loss'] < eg['final_val_loss'] else "egroll",
            "val_loss_delta": bp['final_val_loss'] - eg['final_val_loss'],
            "val_ppl_delta": bp['final_val_ppl'] - eg['final_val_ppl'],
        }
        with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {args.output_dir}/comparison.json")

    elif 'egroll' in results:
        print(f"\nEGGROLL: val_loss={results['egroll']['final_val_loss']:.4f} "
              f"| val_ppl={results['egroll']['final_val_ppl']:.2f}")
    elif 'backprop' in results:
        print(f"\nBackprop: val_loss={results['backprop']['final_val_loss']:.4f} "
              f"| val_ppl={results['backprop']['final_val_ppl']:.2f}")

    print(f"\nTrackio: http://127.0.0.1:7861/?project=hybrid-slm")
    print("=" * 70)


if __name__ == "__main__":
    main()
