"""訓練 Model A routability surrogate。

讀 gen_data 產出的 npz（x/edge_index/raster/routed_fraction）→ 以 netlist_id 切分防洩漏
→ 訓練 → 報 MAE + 同 group 內 Spearman ρ（驗收門檻 ρ≥0.85，placement-routing-checker §5）。

用法：python -m genpcb.surrogate.train --data data/surrogate_data --epochs 60
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from genpcb.surrogate.model import RoutabilitySurrogate


def load_samples(data_dir: str) -> list[dict]:
    out = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".npz"):
            continue
        s = np.load(os.path.join(data_dir, fn))
        rf = float(s["routed_fraction"])
        if rf < 0:                                  # -1 = 標註失敗，跳過
            continue
        fam, seed, variant = fn[:-4].split("_")
        out.append({
            "x": torch.tensor(s["x"], dtype=torch.float32),
            "edge_index": torch.tensor(s["edge_index"], dtype=torch.long),
            "raster": torch.tensor(s["raster"], dtype=torch.float32),
            "y": rf, "netlist_id": f"{fam}_{seed}",
        })
    return out


def split_by_netlist(samples, val_frac=0.2, seed=17):
    sigs = sorted({s["netlist_id"] for s in samples})
    random.Random(seed).shuffle(sigs)
    n_val = max(1, int(len(sigs) * val_frac))
    val_sigs = set(sigs[:n_val])
    tr = [s for s in samples if s["netlist_id"] not in val_sigs]
    va = [s for s in samples if s["netlist_id"] in val_sigs]
    return tr, va


def _rank(a):
    order = np.argsort(a)
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    return r


def spearman(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def grouped_spearman(samples, preds) -> float:
    """同 netlist group 內排序相關，再跨 group 平均（對齊 GRPO group-relative）。"""
    by = {}
    for s, p in zip(samples, preds):
        by.setdefault(s["netlist_id"], []).append((s["y"], p))
    rs = []
    for pairs in by.values():
        if len(pairs) >= 2:
            ys, ps = zip(*pairs)
            r = spearman(ys, ps)
            if not np.isnan(r):
                rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def evaluate(model, samples):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in samples:
            preds.append(float(model(s["x"], s["edge_index"], s["raster"])))
    y = np.array([s["y"] for s in samples])
    p = np.array(preds)
    mae = float(np.mean(np.abs(y - p)))
    return mae, spearman(y, p), grouped_spearman(samples, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/surrogate_data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default="experiments/surrogate_a.pt")
    args = ap.parse_args()

    samples = load_samples(args.data)
    tr, va = split_by_netlist(samples)
    print(f"[surrogate] {len(samples)} samples ({len({s['netlist_id'] for s in samples})} netlists) "
          f"→ train {len(tr)} / val {len(va)}")
    if not tr or not va:
        print("資料不足（需多個 netlist 才能切分）"); return

    torch.manual_seed(17)
    model = RoutabilitySurrogate(hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(tr)
        opt.zero_grad()
        total = 0.0
        for i, s in enumerate(tr, 1):
            pred = model(s["x"], s["edge_index"], s["raster"])
            loss = loss_fn(pred, torch.tensor(s["y"], dtype=torch.float32))
            loss.backward()
            total += float(loss)
            if i % 8 == 0 or i == len(tr):          # mini-batch via 梯度累積
                opt.step(); opt.zero_grad()
        if ep % 5 == 0 or ep == 1:
            mae, sp, gsp = evaluate(model, va)
            print(f"ep{ep:3d} train_mse {total/len(tr):.4f} | val MAE {mae:.3f} "
                  f"Spearman {sp:.3f} grouped {gsp:.3f}", flush=True)
            if mae < best:
                best = mae
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save(model.state_dict(), args.out)
    print(f"[surrogate] best val MAE {best:.3f} → {args.out}")
    print("驗收：grouped Spearman ≥ 0.85 才准進 GRPO 迴圈（樣本多時才可靠）")


if __name__ == "__main__":
    main()
