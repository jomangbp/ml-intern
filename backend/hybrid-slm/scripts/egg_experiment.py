"""
EGGROLL Experiment: Faithful Reproduction + Comparison
======================================================

Based on arXiv:2511.16652 "Evolution Strategies at the Hyperscale"

The paper trains:
- EGG RNN: 6-layer, 256-dim, ~4.86M params, int8-only
- Dataset: MiniPile (JeanKaddour/minipile), character-level (UTF-8 bytes)
- Metric: bits/byte
- Population: up to 2^20, rank=1, antithetic sampling

This script runs a fair comparison on a single RTX 3060 (6GB):
1. EGG RNN trained with EGGROLL (ES, grad-free)
2. EGG RNN trained with standard backprop (AdamW)

Both use the same:
- Architecture (EGG RNN, 6L-256D)
- Data (MiniPile, character-level)
- Compute budget (same wall-clock time or same # of seen tokens)
- Evaluation metric (bits/byte on test set)

Usage:
    python scripts/egg_experiment.py --method both --max-tokens 10m
    python scripts/egg_experiment.py --method egroll --population 1024
    python scripts/egg_experiment.py --method backprop
"""

import os
import sys
import math
import time
import gc
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import numpy as np

import trackio

sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════
# EGG RNN Architecture (from the paper)
# ══════════════════════════════════════════════════════════════════

class MinGRUCell(nn.Module):
    """
    Minimal GRU cell (from the paper: modified minGRU).
    
    Simplified GRU with only a reset-like gate:
    h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
    
    where z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)
          \tilde{h}_t = tanh(W_h @ x_t + U_h @ h_{t-1} + b_h)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Gate projections
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=True)
    
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Single step: x_t [B, input_dim], h_prev [B, hidden_dim]"""
        combined = torch.cat([x_t, h_prev], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        h_tilde = torch.tanh(self.W_h(combined))
        h = (1 - z) * h_prev + z * h_tilde
        return h


class EGGRNNLayer(nn.Module):
    """Single EGG RNN layer: LayerNorm → minGRU → LayerNorm → MLP"""
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        mlp_dim = int(hidden_dim * mlp_ratio)
        
        # Pre-LayerNorm
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # minGRU
        self.gru = MinGRUCell(hidden_dim, hidden_dim)
        
        # Post-LayerNorm
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # MLP
        self.mlp_up = nn.Linear(hidden_dim, mlp_dim)
        self.mlp_down = nn.Linear(mlp_dim, hidden_dim)
    
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        """x_t [B, D], h_prev [B, D] -> (output [B, D], new_h [B, D])"""
        # GRU with pre-norm
        h_normed = self.ln1(h_prev)
        h_new = self.gru(x_t, h_normed)
        
        # MLP with pre-norm + residual
        mlp_in = self.ln2(h_new)
        mlp_out = self.mlp_down(F.relu(self.mlp_up(mlp_in)))
        out = h_new + mlp_out
        
        return out, h_new


class EGGModel(nn.Module):
    """
    EGG (Evolved Generative) RNN
    
    From arXiv:2511.16652:
    - 6 layers, hidden_dim=256
    - ~4.86M parameters
    - Character-level (vocab_size=256)
    - Decoder-only: predicts next character
    
    Architecture:
        embed(x) -> [Layer1 GRU -> Layer2 GRU -> ... -> Layer6 GRU] -> head -> logits
    """
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 256, 
                 num_layers: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        
        # RNN layers
        self.layers = nn.ModuleList([
            EGGRNNLayer(hidden_dim, mlp_ratio) for _ in range(num_layers)
        ])
        
        # Output head
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Initialize
        self.apply(self._init_weights)
        self._count_params()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def _count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"EGG RNN: {n:,} params ({n/1e6:.2f}M)")
    
    def forward(self, input_ids: torch.Tensor, 
                states: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [B, L] character IDs (0-255)
            states: list of [B, D] hidden states per layer (None = zeros)
        
        Returns:
            dict with 'logits' [B, L, vocab_size] and 'states'
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Initialize hidden states
        if states is None:
            states = [torch.zeros(B, self.hidden_dim, device=device) 
                      for _ in range(self.num_layers)]
        
        all_logits = []
        
        # Sequential token processing (RNN)
        for t in range(L):
            x_t = self.embed(input_ids[:, t])  # [B, D]
            
            # Process through layers
            new_states = []
            for i, layer in enumerate(self.layers):
                x_t, h_new = layer(x_t, states[i])
                new_states.append(h_new)
            states = new_states
            
            # Predict next token
            logits_t = self.head(x_t)  # [B, vocab_size]
            all_logits.append(logits_t)
        
        logits = torch.stack(all_logits, dim=1)  # [B, L, vocab_size]
        return {'logits': logits, 'states': states}
    
    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute character-level cross-entropy loss."""
        outputs = self.forward(input_ids)
        logits = outputs['logits']
        
        # Shift: predict next character
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1)
        )
        return loss


# ══════════════════════════════════════════════════════════════════
# Dataset: MiniPile character-level
# ══════════════════════════════════════════════════════════════════

class CharacterDataset(Dataset):
    """
    Character-level dataset from raw text.
    Encodes text as UTF-8 bytes (vocab_size=256).
    """
    def __init__(self, texts: List[str], seq_length: int = 256):
        self.seq_length = seq_length
        # Encode all text as bytes
        all_bytes = bytearray()
        for text in texts:
            encoded = text.encode('utf-8', errors='replace')
            all_bytes.extend(encoded)
            all_bytes.append(0)  # Document separator
        
        self.data = torch.tensor(list(all_bytes), dtype=torch.long)
        self.num_sequences = max(0, (len(self.data) - seq_length - 1))
        
        print(f"CharacterDataset: {len(self.data):,} bytes, "
              f"{self.num_sequences:,} sequences of length {seq_length}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx
        end = start + self.seq_length + 1
        chunk = self.data[start:end]
        return {'input_ids': chunk[:-1], 'labels': chunk[1:]}
    
    def bits_per_byte(self, model, device='cuda', max_batches=100, batch_size=16):
        """Evaluate bits/byte on this dataset."""
        loader = DataLoader(self, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
        model.eval()
        total_bytes = 0
        total_bits = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                input_ids = batch['input_ids'].to(device)
                outputs = model(input_ids)
                logits = outputs['logits']
                
                # Cross-entropy in nats
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = input_ids[:, 1:].contiguous()
                
                # Per-byte log-likelihood
                log_probs = F.log_softmax(shift_logits, dim=-1)
                target_log_probs = log_probs.gather(
                    -1, shift_targets.unsqueeze(-1)
                ).squeeze(-1)
                
                total_bits += (-target_log_probs.sum() / math.log(2)).item()
                total_bytes += shift_targets.numel()
                
                del input_ids, outputs, logits
        
        bits_byte = total_bits / total_bytes if total_bytes > 0 else float('inf')
        return bits_byte


def load_minipile(seq_length=256, max_docs_train=10000, max_docs_test=500):
    """Load MiniPile dataset for character-level training."""
    from datasets import load_dataset
    
    print("Loading MiniPile dataset...")
    ds = load_dataset("JeanKaddour/minipile", trust_remote_code=True)
    
    train_texts = [item['text'] for item in ds['train'].select(range(min(max_docs_train, len(ds['train']))))]
    test_texts = [item['text'] for item in ds['test'].select(range(min(max_docs_test, len(ds['test']))))]
    
    print(f"  Train docs: {len(train_texts)}")
    print(f"  Test docs: {len(test_texts)}")
    
    train_dataset = CharacterDataset(train_texts, seq_length=seq_length)
    test_dataset = CharacterDataset(test_texts, seq_length=seq_length)
    
    return train_dataset, test_dataset


# ══════════════════════════════════════════════════════════════════
# EGGROLL Training (ES with low-rank perturbations)
# ══════════════════════════════════════════════════════════════════

class EGGROLLTrainer:
    """
    EGGROLL trainer following arXiv:2511.16652.
    
    Key algorithm:
    1. Sample N pairs of antithetic perturbations: ε and -ε
    2. Evaluate fitness f(θ+σε) and f(θ-σε)
    3. Score: s = sign(f(θ+σε) - f(θ-σε)) ∈ {-1, 0, +1}
    4. Update: θ += α * s * ε  (rank-1 perturbation)
    
    Uses rank-1 perturbations: E = ab^T where a∈R^m, b∈R^n
    """
    
    def __init__(self, model, population_size=1024, noise_scale=0.0625,
                 learning_rate=0.01, device='cuda'):
        self.model = model
        self.population_size = population_size
        self.noise_scale = noise_scale  # σ = 2^(-4) = 0.0625 from paper
        self.learning_rate = learning_rate
        self.device = device
        
        # Cache parameter structure
        self.param_data = []
        for p in model.parameters():
            if p.requires_grad:
                self.param_data.append(p)
    
    def _sample_rank1(self, shape):
        """Sample rank-1 perturbation: E = ab^T / ||a|| / ||b||"""
        m = shape[0]
        n = int(np.prod(shape[1:]))
        a = torch.randn(m, device=self.device)
        b = torch.randn(n, device=self.device)
        E = torch.outer(a, b).view(shape)
        # Normalize
        E = E / (a.norm() * b.norm() + 1e-8)
        return E
    
    def _fitness(self, batch):
        """Negative CE loss (higher = better)"""
        input_ids = batch['input_ids'].to(self.device)
        with torch.no_grad():
            loss = self.model.compute_loss(input_ids)
            if torch.isnan(loss) or torch.isinf(loss):
                return -20.0
            return -loss.item()
    
    def step(self, batch):
        """
        Single EGGROLL step with antithetic sampling (from paper).
        
        For N/2 antithetic pairs:
            ε_i = rank-1 perturbation
            f_plus = f(θ + σε)
            f_minus = f(θ - σε)
            s_i = sign(f_plus - f_minus)
            θ += α/(N/2) * Σ s_i * ε_i
        """
        n_pairs = self.population_size // 2
        
        # Snapshot params
        original = [p.data.clone() for p in self.param_data]
        
        # Collect antithetic pairs
        perturbations = []
        scores = []
        
        for _ in range(n_pairs):
            # Generate perturbation
            eps = [self._sample_rank1(p.shape) for p in self.param_data]
            
            # Evaluate positive perturbation
            for i, p in enumerate(self.param_data):
                p.data.copy_(original[i] + self.noise_scale * eps[i])
            f_plus = self._fitness(batch)
            
            # Evaluate negative perturbation
            for i, p in enumerate(self.param_data):
                p.data.copy_(original[i] - self.noise_scale * eps[i])
            f_minus = self._fitness(batch)
            
            # Score: sign-based (from paper)
            s = 1.0 if f_plus > f_minus else (-1.0 if f_plus < f_minus else 0.0)
            
            perturbations.append(eps)
            scores.append(s)
        
        # Restore original
        for i, p in enumerate(self.param_data):
            p.data.copy_(original[i])
        
        # Apply update
        with torch.no_grad():
            for i, p in enumerate(self.param_data):
                update = sum(scores[j] * perturbations[j][i] 
                            for j in range(n_pairs))
                p.data.add_(self.learning_rate * update / n_pairs)
        
        # Report estimated loss from positive evaluations
        return {
            'score_pos': sum(1 for s in scores if s > 0),
            'score_neg': sum(1 for s in scores if s < 0),
            'score_tie': sum(1 for s in scores if s == 0),
        }


# ══════════════════════════════════════════════════════════════════
# Backprop Training (AdamW)
# ══════════════════════════════════════════════════════════════════

class BackpropTrainer:
    """Standard AdamW training for comparison."""
    
    def __init__(self, model, lr=3e-4, warmup=500, max_steps=5000, device='cuda'):
        self.model = model
        self.device = device
        self.max_steps = max_steps
        self.warmup = warmup
        self.base_lr = lr
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                               betas=(0.9, 0.95))
    
    def step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.model.compute_loss(input_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def get_lr(self, step):
        if step < self.warmup:
            return self.base_lr * step / max(self.warmup, 1)
        progress = (step - self.warmup) / max(self.max_steps - self.warmup, 1)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════
# Main Experiment
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EGGROLL vs Backprop on EGG RNN")
    parser.add_argument("--method", choices=["egroll", "backprop", "both"],
                        default="both")
    parser.add_argument("--population", type=int, default=1024,
                        help="EGGROLL population size (N/2 antithetic pairs)")
    parser.add_argument("--noise-scale", type=float, default=0.0625,
                        help="Perturbation σ (paper: 2^-4 = 0.0625)")
    parser.add_argument("--egroll-lr", type=float, default=0.01,
                        help="EGGROLL learning rate")
    parser.add_argument("--backprop-lr", type=float, default=3e-4)
    parser.add_argument("--seq-length", type=int, default=256,
                        help="Sequence length in characters")
    parser.add_argument("--max-tokens", type=str, default="10m",
                        help="Total tokens to train on (e.g., 10m, 50m)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--max-docs", type=int, default=50000,
                        help="Max MiniPile documents to load")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/egg-experiment")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    args = parser.parse_args()
    
    # Parse max_tokens
    mt = args.max_tokens.lower()
    if mt.endswith('m'):
        max_tokens = int(float(mt[:-1]) * 1e6)
    elif mt.endswith('k'):
        max_tokens = int(float(mt[:-1]) * 1e3)
    else:
        max_tokens = int(mt)
    
    print("\n" + "=" * 70)
    print("  EGGROLL vs BACKPROP — EGG RNN on MiniPile (character-level)")
    print("=" * 70)
    print(f"  Model:        EGG RNN ({args.num_layers}L-{args.hidden_dim}D)")
    print(f"  Dataset:      MiniPile (character-level, vocab=256)")
    print(f"  Seq length:   {args.seq_length} chars")
    print(f"  Max tokens:   {max_tokens:,}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Method:       {args.method}")
    if args.method in ('egroll', 'both'):
        print(f"  Population:   {args.population} ({args.population//2} antithetic pairs)")
        print(f"  Noise σ:      {args.noise_scale}")
        print(f"  ES LR:        {args.egroll_lr}")
    if args.method in ('backprop', 'both'):
        print(f"  Backprop LR:  {args.backprop_lr}")
    print(f"  Output:       {args.output_dir}")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_dataset, test_dataset = load_minipile(
        seq_length=args.seq_length,
        max_docs_train=args.max_docs,
        max_docs_test=500,
    )
    
    # Trackio
    trackio.init(
        project="hybrid-slm",
        name="egg-egroll-vs-backprop",
        config={
            "experiment": "egg-egroll-vs-backprop",
            "model": f"EGG RNN ({args.num_layers}L-{args.hidden_dim}D)",
            "dataset": "MiniPile (char-level)",
            "max_tokens": max_tokens,
            "population_size": args.population,
            "noise_scale": args.noise_scale,
            "egroll_lr": args.egroll_lr,
            "backprop_lr": args.backprop_lr,
            "seq_length": args.seq_length,
            "batch_size": args.batch_size,
        },
    )
    
    results = {}
    
    # ── EGGROLL Training ────────────────────────────────────────
    if args.method in ('egroll', 'both'):
        print("\n" + "=" * 60)
        print("EGGROLL TRAINING (gradient-free)")
        print("=" * 60)
        
        model_eg = EGGModel(
            vocab_size=256, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).cuda()
        
        trainer = EGGROLLTrainer(
            model_eg,
            population_size=args.population,
            noise_scale=args.noise_scale,
            learning_rate=args.egroll_lr,
            device='cuda',
        )
        
        eg_dir = os.path.join(args.output_dir, "egroll")
        os.makedirs(eg_dir, exist_ok=True)
        log_file = open(os.path.join(eg_dir, "training_log.txt"), "w")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        train_iter = iter(train_loader)
        
        start_time = time.time()
        tokens_seen = 0
        step = 0
        
        while tokens_seen < max_tokens:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            metrics = trainer.step(batch)
            step += 1
            tokens_seen += batch['input_ids'].numel()
            
            # Eval
            if step % args.eval_every == 0:
                bpb = test_dataset.bits_per_byte(model_eg, max_batches=50, batch_size=args.batch_size)
                elapsed = time.time() - start_time
                tok_s = tokens_seen / elapsed if elapsed > 0 else 0
                
                line = (f"step={step} | tokens={tokens_seen:,} | "
                       f"bits/byte={bpb:.4f} | tok/s={tok_s:,.0f} | "
                       f"wins={metrics['score_pos']} losses={metrics['score_neg']} ties={metrics['score_tie']}")
                print(line)
                log_file.write(line + "\n")
                log_file.flush()
                
                trackio.log({
                    "egroll/bits_per_byte": bpb,
                    "egroll/tokens_seen": tokens_seen,
                    "egroll/tokens_per_sec": tok_s,
                    "egroll/wins": metrics['score_pos'],
                    "egroll/losses": metrics['score_neg'],
                    "egroll/step": step,
                })
                
                # Save best
                torch.save(model_eg.state_dict(), 
                          os.path.join(eg_dir, f"checkpoint-{step}.pt"))
            
            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final eval
        final_bpb = test_dataset.bits_per_byte(model_eg, max_batches=200, batch_size=args.batch_size)
        elapsed = time.time() - start_time
        
        print(f"\nEGGROLL Final: {final_bpb:.4f} bits/byte | {tokens_seen:,} tokens | {elapsed/60:.1f} min")
        torch.save(model_eg.state_dict(), os.path.join(eg_dir, "final.pt"))
        
        results['egroll'] = {
            'method': 'egroll',
            'final_bpb': final_bpb,
            'tokens_seen': tokens_seen,
            'steps': step,
            'population': args.population,
            'noise_scale': args.noise_scale,
            'lr': args.egroll_lr,
            'training_time_min': elapsed / 60,
        }
        
        log_file.close()
        del model_eg, trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    # ── Backprop Training ───────────────────────────────────────
    if args.method in ('backprop', 'both'):
        print("\n" + "=" * 60)
        print("BACKPROP TRAINING (AdamW)")
        print("=" * 60)
        
        model_bp = EGGModel(
            vocab_size=256, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).cuda()
        
        max_steps = max_tokens // (args.batch_size * args.seq_length)
        
        trainer = BackpropTrainer(
            model_bp, lr=args.backprop_lr, warmup=max(100, max_steps // 10),
            max_steps=max_steps, device='cuda',
        )
        
        bp_dir = os.path.join(args.output_dir, "backprop")
        os.makedirs(bp_dir, exist_ok=True)
        log_file = open(os.path.join(bp_dir, "training_log.txt"), "w")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        train_iter = iter(train_loader)
        
        start_time = time.time()
        tokens_seen = 0
        best_bpb = float('inf')
        
        for step in range(1, max_steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            loss = trainer.step(batch)
            tokens_seen += batch['input_ids'].numel()
            
            # Update LR
            lr = trainer.get_lr(step)
            for pg in trainer.optimizer.param_groups:
                pg['lr'] = lr
            
            # Eval
            if step % args.eval_every == 0:
                bpb = test_dataset.bits_per_byte(model_bp, max_batches=50, batch_size=args.batch_size)
                elapsed = time.time() - start_time
                tok_s = tokens_seen / elapsed if elapsed > 0 else 0
                
                line = (f"step={step} | tokens={tokens_seen:,} | "
                       f"loss={loss:.4f} | bits/byte={bpb:.4f} | "
                       f"lr={lr:.2e} | tok/s={tok_s:,.0f}")
                print(line)
                log_file.write(line + "\n")
                log_file.flush()
                
                trackio.log({
                    "backprop/loss": loss,
                    "backprop/bits_per_byte": bpb,
                    "backprop/tokens_seen": tokens_seen,
                    "backprop/tokens_per_sec": tok_s,
                    "backprop/learning_rate": lr,
                    "backprop/step": step,
                })
                
                if bpb < best_bpb:
                    best_bpb = bpb
                    torch.save(model_bp.state_dict(),
                              os.path.join(bp_dir, "best.pt"))
        
        # Final eval
        final_bpb = test_dataset.bits_per_byte(model_bp, max_batches=200, batch_size=args.batch_size)
        elapsed = time.time() - start_time
        
        print(f"\nBackprop Final: {final_bpb:.4f} bits/byte | {tokens_seen:,} tokens | {elapsed/60:.1f} min")
        torch.save(model_bp.state_dict(), os.path.join(bp_dir, "final.pt"))
        
        results['backprop'] = {
            'method': 'backprop',
            'final_bpb': final_bpb,
            'best_bpb': best_bpb,
            'tokens_seen': tokens_seen,
            'steps': max_steps,
            'lr': args.backprop_lr,
            'training_time_min': elapsed / 60,
        }
        
        log_file.close()
        del model_bp, trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    # ── Comparison ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)
    
    if len(results) == 2:
        eg = results['egroll']
        bp = results['backprop']
        
        print(f"\n{'Metric':<25} {'EGGROLL':>12} {'Backprop':>12}")
        print("-" * 50)
        print(f"{'Final bits/byte':<25} {eg['final_bpb']:>12.4f} {bp['final_bpb']:>12.4f}")
        print(f"{'Tokens seen':<25} {eg['tokens_seen']:>12,} {bp['tokens_seen']:>12,}")
        print(f"{'Steps':<25} {eg['steps']:>12,} {bp['steps']:>12,}")
        print(f"{'Time (min)':<25} {eg['training_time_min']:>12.1f} {bp['training_time_min']:>12.1f}")
        
        winner = "EGGROLL" if eg['final_bpb'] < bp['final_bpb'] else "Backprop"
        delta = abs(eg['final_bpb'] - bp['final_bpb'])
        print(f"\n  Winner: {winner} (Δ = {delta:.4f} bits/byte)")
        
        paper_bp = 3.58  # From paper: backprop Transformer baseline
        paper_eg = 3.40  # From paper: EGGROLL best
        print(f"\n  Paper reference: Backprop={paper_bp} bits/byte, EGGROLL={paper_eg} bits/byte")
    
    # Save comparison
    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Trackio: http://127.0.0.1:7861/?project=hybrid-slm")


if __name__ == "__main__":
    main()
