#!/usr/bin/env python3
"""
LeWM training v6 — exact le-wm recipe, scaled for small GPU.
Key: encode ALL frames together (no stop-grad), SIGReg only regularization.
Matches the paper's lejepa_forward pattern.
"""

import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

LEWM_PATH = Path("/home/jgbla/repos/le-wm")
sys.path.insert(0, str(LEWM_PATH))
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg


class EpisodeAwareDataset(Dataset):
    def __init__(self, h5_path, context_len=3, num_preds=1, img_size=128):
        import h5py
        self.ctx, self.np, self.img = context_len, num_preds, img_size
        with h5py.File(h5_path, "r") as f:
            self.adim = f.attrs["action_dim"]
            oh, ow = f.attrs.get("img_height", f["pixels"].shape[1]), f.attrs.get("img_width", f["pixels"].shape[2])
            self.pixels = f["pixels"][:]
            self.actions = f["action"][:]
            eps = f.get("ep_start", np.array([0], dtype=np.int64))[:]
        if oh != img_size or ow != img_size:
            res = np.zeros((self.pixels.shape[0], img_size, img_size, 3), dtype=np.uint8)
            for i in range(self.pixels.shape[0]):
                res[i] = np.array(Image.fromarray(self.pixels[i]).resize((img_size, img_size), Image.BILINEAR))
            self.pixels = res
        self._eps = eps
        self.windows = []
        for ei in range(len(eps)):
            b = int(eps[ei]); e = int(eps[ei+1]) if ei+1 < len(eps) else self.pixels.shape[0]
            for o in range(e - b - self.ctx - self.np + 1):
                self.windows.append(b + o)
        self.neps = len(eps)
        lens = [int(eps[i+1]-eps[i]) if i+1<len(eps) else self.pixels.shape[0]-int(eps[i]) for i in range(len(eps))]
        print(f"  {len(self.windows)} samples, {self.neps} eps, ctx={self.ctx}, np={self.np}, "
              f"len: min={min(lens)} max={max(lens)} mean={np.mean(lens):.1f}")

    def __len__(self): return len(self.windows)
    def __getitem__(self, i):
        s = self.windows[i]; e = s + self.ctx + self.np
        p = torch.from_numpy(self.pixels[s:e]).float()/255.0
        a = torch.from_numpy(self.actions[s:e-1]).float()
        return {"pixels": p.permute(0,3,1,2), "action": a}


def split_eps(ds, frac, seed):
    eps = list(range(ds.neps)); np.random.RandomState(seed).shuffle(eps)
    trs = set(eps[:max(1,int(frac*ds.neps))])
    ti, vi = [], []
    for i, sf in enumerate(ds.windows):
        ei = max(0, np.searchsorted(ds._eps, sf, side='right')-1)
        (ti if ei in trs else vi).append(i)
    return torch.utils.data.Subset(ds, ti), torch.utils.data.Subset(ds, vi)


def build_lewm(img=128, ps=14, ed=192, hs=3, ad=5, es="tiny", pd=6, ph=16, pm=2048, do=0.1):
    from transformers import ViTConfig, ViTModel
    sc = {"tiny":{"hidden_size":192,"num_hidden_layers":12,"num_attention_heads":3},
          "small":{"hidden_size":384,"num_hidden_layers":12,"num_attention_heads":6}}[es]
    vc = ViTConfig(image_size=img, patch_size=ps, hidden_size=sc["hidden_size"],
                   num_hidden_layers=sc["num_hidden_layers"],
                   num_attention_heads=sc["num_attention_heads"],
                   intermediate_size=sc["hidden_size"]*4)
    enc = ViTModel(vc); hd = sc["hidden_size"]
    pred = ARPredictor(num_frames=hs, input_dim=ed, hidden_dim=hd, output_dim=hd,
                       depth=pd, heads=ph, mlp_dim=pm, dim_head=64, dropout=do, emb_dropout=0.0)
    ae = Embedder(input_dim=ad, smoothed_dim=ad, emb_dim=ed, mlp_scale=4)
    proj = MLP(input_dim=hd, output_dim=ed, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    pproj = MLP(input_dim=hd, output_dim=ed, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    return JEPA(encoder=enc, predictor=pred, action_encoder=ae, projector=proj, pred_proj=pproj)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data"); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=8); p.add_argument("--ga", type=int, default=1)
    p.add_argument("--img", type=int, default=128); p.add_argument("--ed", type=int, default=192)
    p.add_argument("--ctx", type=int, default=3); p.add_argument("--np", type=int, default=1)
    p.add_argument("--ad", type=int, default=0); p.add_argument("--lambd", type=float, default=0.09)
    p.add_argument("--pd", type=int, default=6); p.add_argument("--do", type=float, default=0.1)
    p.add_argument("--es", default="tiny"); p.add_argument("--out", default="outputs")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--tf", type=float, default=0.9)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)

    import h5py
    with h5py.File(args.data) as f:
        if args.ad == 0: args.ad = f.attrs["action_dim"]
    ds = EpisodeAwareDataset(Path(args.data), args.ctx, args.np, args.img)
    if ds.neps > 1:
        td, vd = split_eps(ds, args.tf, args.seed)
    else:
        n = int(0.8*len(ds)); td, vd = torch.utils.data.random_split(ds, [n, len(ds)-n],
            generator=torch.Generator().manual_seed(args.seed))
    print(f"Train: {len(td)}, Val: {len(vd)}")

    tdl = DataLoader(td, args.bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    vdl = DataLoader(vd, args.bs, shuffle=False, num_workers=2, pin_memory=True)
    eff = args.bs * args.ga
    print(f"Batches: train={len(tdl)}, val={len(vdl)}, eff_bs={eff}")

    model = build_lewm(args.img, ed=args.ed, hs=args.ctx, ad=args.ad, es=args.es,
                       pd=args.pd, do=args.do).to(dev)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")
    sigreg = SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    wu = max(1, args.epochs//20)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(tdl)//args.ga, pct_start=wu/args.epochs,
        div_factor=25.0, final_div_factor=100.0)

    bv = float("inf")
    hist = {"train":[], "val":[], "pred":[], "sig":[]}

    print(f"\n{'='*60}\nLeWM v6  epochs={args.epochs}  lr={args.lr}  λ={args.lambd}  WD={args.wd}")
    print(f"bs={args.bs}×{args.ga}={eff}  img={args.img}  embed={args.ed}  pd={args.pd}  do={args.do}")
    print(f"{'='*60}\n")

    for ep in range(args.epochs):
        model.train()
        tl, pl, sl, nb = 0.0, 0.0, 0.0, 0
        # Per-step LR scheduler
        step_count = 0
        for bi, batch in enumerate(tdl):
            px = batch["pixels"].to(dev)     # (B, T_total, 3, H, W) where T_total = ctx + np
            ac = batch["action"].to(dev)     # (B, T_total-1, action_dim)
            B, T, C, H, W = px.shape  # T = ctx + np

            # ── Exact le-wm pattern: encode ALL frames together ──
            # Flatten: (B*T, C, H, W)
            pf = px.reshape(B*T, C, H, W)
            vo = model.encoder(pf)
            cls = vo.last_hidden_state[:, 0]          # (B*T, hdim)
            emb = model.projector(cls).reshape(B, T, -1)  # (B, T, edim)
            act_emb = model.action_encoder(ac)        # (B, T-1, edim)

            # Context + target split
            ctx_emb = emb[:, :args.ctx]               # (B, ctx, edim)
            ctx_act = act_emb[:, :args.ctx]           # (B, ctx, edim)
            tgt_emb = emb[:, args.np:]                # (B, ctx, edim) — shifted by np

            # Predict
            ph = model.predictor(ctx_emb, ctx_act)    # (B, ctx, hdim)
            pe = model.pred_proj(ph.reshape(B*args.ctx, -1)).reshape(B, args.ctx, -1)

            pred_loss = F.mse_loss(pe, tgt_emb)
            sigreg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + args.lambd * sigreg_loss
            loss = loss / args.ga; loss.backward()

            if (bi+1) % args.ga == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); sch.step()

            tl += loss.item()*args.ga; pl += pred_loss.item(); sl += sigreg_loss.item(); nb += 1

        # Validation
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for batch in vdl:
                px = batch["pixels"].to(dev); ac = batch["action"].to(dev)
                B, T, C, H, W = px.shape
                pf = px.reshape(B*T, C, H, W)
                vo = model.encoder(pf)
                cls = vo.last_hidden_state[:, 0]
                emb = model.projector(cls).reshape(B, T, -1)
                act_emb = model.action_encoder(ac)
                ctx_emb = emb[:, :args.ctx]; ctx_act = act_emb[:, :args.ctx]
                tgt_emb = emb[:, args.np:]
                ph = model.predictor(ctx_emb, ctx_act)
                pe = model.pred_proj(ph.reshape(B*args.ctx, -1)).reshape(B, args.ctx, -1)
                ploss = F.mse_loss(pe, tgt_emb)
                sloss = sigreg(emb.transpose(0, 1))
                vloss += (ploss + args.lambd*sloss).item(); vn += 1

        hist["train"].append(tl/nb); hist["val"].append(vloss/vn)
        hist["pred"].append(pl/nb); hist["sig"].append(sl/nb)

        if (ep+1)%5==0 or ep==0:
            print(f"Ep {ep+1:3d} | train={tl/nb:.4f} pred={pl/nb:.4f} sig={sl/nb:.4f} | val={vloss/vn:.4f} | lr={sch.get_last_lr()[0]:.2e}")

        if vloss/vn < bv:
            bv = vloss/vn
            torch.save({"epoch":ep, "model_state_dict":model.state_dict(), "val_loss":bv, "config":vars(args)}, od/"best_model.pt")

    torch.save({"epoch":args.epochs-1, "model_state_dict":model.state_dict(), "config":vars(args)}, od/"final_model.pt")
    with open(od/"history.json","w") as f: json.dump(hist, f)
    print(f"\nDone! Best val={bv:.4f}")


if __name__ == "__main__":
    main()
