#!/usr/bin/env python3
"""
Inference / evaluation for trained LeWorldModel computer-use agent.

Demonstrates latent planning:
  1. Encode current screenshot → z_cur
  2. Encode goal screenshot → z_g 
  3. Sample action candidates in latent space
  4. Rollout predictor autoregressively
  5. Select action that minimizes ||ẑ_H - z_g||²

This is the same Cross-Entropy Method (CEM) used in the LeWM paper,
adapted for computer-use action space.

Usage:
    # Evaluate on validation data
    python scripts/eval_lewm.py --checkpoint outputs/best_model.pt --data data/gui_trajectories.h5
    
    # Interactive demo (requires GUI)
    python scripts/eval_lewm.py --checkpoint outputs/best_model.pt --interactive
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LEWM_PATH = Path("/home/jgbla/repos/le-wm")
sys.path.insert(0, str(LEWM_PATH))
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg


def load_model(checkpoint_path: str, device: torch.device, **kwargs) -> JEPA:
    """Load trained LeWM from checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent))
    from train_lewm import build_lewm
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    
    # Override with kwargs
    img_size = kwargs.get("img_size", cfg.get("img_size", 128))
    embed_dim = kwargs.get("embed_dim", cfg.get("embed_dim", 192))
    ctx_len = kwargs.get("ctx_len", cfg.get("ctx_len", 3))
    action_dim = kwargs.get("action_dim", cfg.get("action_dim", 5))
    encoder_scale = kwargs.get("encoder_scale", cfg.get("encoder_scale", "tiny"))
    predictor_depth = cfg.get("predictor_depth", cfg.get("depth", 6))
    predictor_dropout = cfg.get("dropout", cfg.get("predictor_dropout", 0.1))
    
    model = build_lewm(
        img_size=img_size,
        embed_dim=embed_dim,
        history_size=ctx_len,
        action_dim=action_dim,
        encoder_scale=encoder_scale,
        predictor_depth=predictor_depth,
        predictor_dropout=predictor_dropout,
    )
    
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}")
    
    return model


@torch.no_grad()
def encode_screenshot(model: JEPA, pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Encode a screenshot to latent embedding.
    
    Args:
        pixels: (1, 3, H, W) or (B, 3, H, W) normalized [0,1]
    Returns:
        z: (1, embed_dim) or (B, embed_dim)
    """
    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)
    
    pixels = pixels.to(device)
    vit_output = model.encoder(pixels)
    cls_tokens = vit_output.last_hidden_state[:, 0]
    z = model.projector(cls_tokens)
    return z


@torch.no_grad()
def latent_rollout(
    model: JEPA,
    z_start: torch.Tensor,       # (B, embed_dim)
    action_sequence: torch.Tensor, # (B, T, action_dim)
    device: torch.device,
) -> torch.Tensor:
    """Roll out predictor autoregressively in latent space.
    
    Args:
        z_start: starting latent state
        action_sequence: proposed actions (B, T_pred, action_dim)
    Returns:
        z_final: predicted latent state after T_pred steps (B, embed_dim)
    """
    B = z_start.shape[0]
    T_pred = action_sequence.shape[1]
    
    # Start with z_start as context, expand to (B, ctx_len, embed_dim)
    # For simplicity, repeat z_start to fill context
    ctx_len = 3  # match training
    z_seq = z_start.unsqueeze(1).expand(B, ctx_len, -1)  # (B, ctx_len, embed_dim)
    
    for t in range(T_pred):
        # Get current context (last ctx_len states)
        z_ctx = z_seq[:, -ctx_len:]  # (B, ctx_len, embed_dim)
        
        # Get action for this step
        act = action_sequence[:, t:t+1, :]  # (B, 1, action_dim)
        act_emb = model.action_encoder(act)  # (B, 1, embed_dim)
        
        # Pad action context
        act_ctx = torch.cat([
            torch.zeros(B, ctx_len - 1, act_emb.shape[-1], device=device),
            act_emb,
        ], dim=1)  # (B, ctx_len, embed_dim)
        
        # Predict next
        pred_h = model.predictor(z_ctx, act_ctx)  # (B, ctx_len, hidden_dim)
        z_next_h = pred_h[:, -1]  # (B, hidden_dim)
        z_next = model.pred_proj(z_next_h)  # (B, embed_dim)
        
        # Append to sequence
        z_seq = torch.cat([z_seq, z_next.unsqueeze(1)], dim=1)
    
    return z_seq[:, -1]  # final latent state


def cross_entropy_method(
    model: JEPA,
    z_current: torch.Tensor,
    z_goal: torch.Tensor,
    device: torch.device,
    n_samples: int = 100,
    n_iterations: int = 10,
    n_elite: int = 10,
    planning_horizon: int = 5,
    action_dim: int = 5,
    initial_std: float = 0.5,
) -> torch.Tensor:
    """
    Cross-Entropy Method for latent-space planning.
    
    Returns:
        best_action: (planning_horizon, action_dim) — first action to execute
    """
    B = z_current.shape[0]
    
    # Initialize sampling distribution
    mu = torch.zeros(planning_horizon, action_dim, device=device)
    sigma = torch.ones(planning_horizon, action_dim, device=device) * initial_std
    
    best_sequence = None
    best_cost = float("inf")
    
    for iteration in range(n_iterations):
        # Sample action sequences
        action_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
            n_samples, planning_horizon, action_dim, device=device
        )  # (n_samples, H, action_dim)
        
        # Expand for batch
        z_start_expanded = z_current.unsqueeze(1).expand(B, n_samples, -1)
        z_start_flat = z_start_expanded.reshape(B * n_samples, -1)
        action_flat = action_samples.unsqueeze(0).expand(B, n_samples, planning_horizon, action_dim)
        action_flat = action_flat.reshape(B * n_samples, planning_horizon, action_dim)
        
        # Rollout
        z_final = latent_rollout(model, z_start_flat, action_flat, device)
        
        # Cost: distance to goal
        z_goal_expanded = z_goal.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        cost = F.mse_loss(z_final, z_goal_expanded, reduction="none").sum(dim=-1)
        cost = cost.reshape(B, n_samples)  # (B, n_samples)
        
        # Select elite samples (per batch)
        avg_cost = cost.mean(dim=0)  # (n_samples,)
        elite_indices = avg_cost.topk(n_elite, largest=False).indices
        
        elite_actions = action_samples[elite_indices]  # (n_elite, H, action_dim)
        
        # Update distribution
        mu = elite_actions.mean(dim=0)
        sigma = elite_actions.std(dim=0) + 1e-6
        
        # Track best
        min_cost, min_idx = avg_cost.min(dim=0)
        if min_cost < best_cost:
            best_cost = min_cost.item()
            best_sequence = action_samples[min_idx]
    
    return best_sequence, best_cost


def evaluate_on_dataset(
    model: JEPA,
    data_path: str,
    device: torch.device,
    num_samples: int = 20,
    img_size: int = 128,
):
    """Evaluate latent prediction on held-out trajectories."""
    from train_lewm import EpisodeAwareGuiDataset
    
    ds = EpisodeAwareGuiDataset(Path(data_path), context_len=3, img_size=img_size)
    
    print(f"\nEvaluating on {min(num_samples, len(ds))} samples...")
    
    pred_errors = []
    latent_norms = []
    
    indices = np.random.choice(len(ds), min(num_samples, len(ds)), replace=False)
    
    for i in indices:
        sample = ds[i]
        
        pixels_cur = sample["pixels"]           # (ctx_len, 3, H, W)
        tgt_pixels = sample["tgt_pixels"]       # (ctx_len, 3, H, W)
        
        # Encode last context frame and its target
        z_cur = encode_screenshot(model, pixels_cur[-1].unsqueeze(0), device)
        z_goal = encode_screenshot(model, tgt_pixels[-1].unsqueeze(0), device)
        
        latent_norms.append(torch.norm(z_cur).item())
        
        # One-step prediction with zero action
        act = torch.zeros(1, 1, ds.action_dim, device=device)
        z_pred = latent_rollout(model, z_cur, act, device)
        
        error = F.mse_loss(z_pred, z_goal).item()
        pred_errors.append(error)
    
    avg_error = np.mean(pred_errors)
    std_error = np.std(pred_errors)
    avg_norm = np.mean(latent_norms)
    
    print(f"  Average latent prediction error: {avg_error:.6f} ± {std_error:.6f}")
    print(f"  Min: {np.min(pred_errors):.6f}, Max: {np.max(pred_errors):.6f}")
    print(f"  Average ||z||: {avg_norm:.3f}")
    print(f"  Relative error (MSE / ||z||²): {avg_error / (avg_norm**2):.4f}")
    
    return avg_error


def interactive_demo(model: JEPA, device: torch.device):
    """Interactive latent planning demo (requires GUI)."""
    try:
        from mss import mss
        from PIL import Image
        
        sct = mss()
        monitor = sct.monitors[1]
        
        print("\n=== Interactive LeWM Computer-Use Demo ===")
        print("Commands: 'capture_current', 'capture_goal', 'plan', 'quit'")
        print("Flow: capture current → capture goal → plan → execute best action")
        
        z_cur = None
        z_goal = None
        
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "quit":
                break
            
            elif cmd.startswith("capture_current"):
                img = sct.grab(monitor)
                arr = np.array(Image.fromarray(np.array(img)[:, :, :3][:, :, ::-1]).resize((128, 128)))
                pixels = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
                z_cur = encode_screenshot(model, pixels, device)
                print(f"  Captured current state → latent dim={z_cur.shape[-1]}")
            
            elif cmd.startswith("capture_goal"):
                img = sct.grab(monitor)
                arr = np.array(Image.fromarray(np.array(img)[:, :, :3][:, :, ::-1]).resize((128, 128)))
                pixels = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
                z_goal = encode_screenshot(model, pixels, device)
                print(f"  Captured goal state")
            
            elif cmd == "plan":
                if z_cur is None or z_goal is None:
                    print("  Capture both states first!")
                    continue
                
                print("  Planning in latent space (CEM)...")
                best_action, cost = cross_entropy_method(
                    model, z_cur, z_goal, device,
                    n_samples=50, n_iterations=5, n_elite=5,
                    planning_horizon=5,
                )
                print(f"  Best cost: {cost:.4f}")
                print(f"  First action: mouse=({best_action[0, 0]:.3f}, {best_action[0, 1]:.3f}), "
                      f"click={int(best_action[0, 2])}, scroll={best_action[0, 3]:.0f}")
            
            else:
                print("  Unknown command")
    
    except ImportError as e:
        print(f"Cannot start interactive demo: {e}")
        print("Requires: mss, pyautogui (GUI environment)")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeWM for Computer Use")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, help="Path to HDF5 for evaluation")
    parser.add_argument("--interactive", action="store_true", help="Interactive latent planning demo")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device, img_size=args.img_size)
    
    # Evaluate
    if args.data:
        evaluate_on_dataset(model, args.data, device, args.num_samples, args.img_size)
    
    if args.interactive:
        interactive_demo(model, device)
    
    if not args.data and not args.interactive:
        print("Nothing to do. Use --data and/or --interactive")


if __name__ == "__main__":
    main()
