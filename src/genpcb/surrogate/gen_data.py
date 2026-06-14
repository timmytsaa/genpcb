"""Surrogate 資料產線：generate → SA anchor → 品質光譜 → 特徵 → 標註 → 存。

每個 netlist 產 8 個品質不同的擺位（同 netlist = 同 group，存 netlist_id 供切分/ranking）。
標註 label_fn 在 Colab/有 Freerouting 的機器傳 routing_reward；標註是慢瓶頸（每板數十秒）。

用法（Colab，有 Freerouting）：
    from genpcb.kicad.route import routing_reward
    from genpcb.surrogate.gen_data import build_dataset
    build_dataset(500, "/content/drive/MyDrive/genpcb/surrogate_data",
                  label_fn=lambda b: routing_reward(b, jar="/content/freerouting.jar"))
"""

from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np

from genpcb.data.perturb import quality_spectrum
from genpcb.data.place_sa import sa_place
from genpcb.data.procedural import FAMILIES, generate_board
from genpcb.surrogate.features import board_to_graph, board_to_raster


def build_dataset(n_netlists: int, out_dir: str, label_fn: Callable, sa_steps: int = 2500,
                  seed0: int = 0, raster_size: int = 64, verbose: bool = True) -> list[dict]:
    """產 n_netlists × 8 個樣本，每個存 .npz（x/edge_index/raster/routed_fraction）+ manifest.jsonl。

    verbose：每板印進度（Freerouting 慢，無進度會像當機）。斷點續跑：已存的 .npz 跳過。
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    total = n_netlists * 8
    done = 0
    if verbose:
        print(f"[build_dataset] {n_netlists} netlists × 8 = {total} 板 → {out_dir}", flush=True)
    for i in range(n_netlists):
        fam = FAMILIES[i % len(FAMILIES)]
        anchor = sa_place(generate_board(fam, seed=seed0 + i), steps=sa_steps, seed=seed0 + i)
        for variant, bv in quality_spectrum(anchor, seed=seed0 + i):
            sid = f"{fam}_{seed0 + i}_{variant}"
            npz_path = os.path.join(out_dir, sid + ".npz")
            done += 1
            if os.path.exists(npz_path):                      # 斷點續跑：只跳過「成功」的，失敗(-1)重試
                prev = float(np.load(npz_path)["routed_fraction"])
                if prev >= 0:
                    manifest.append({
                        "id": sid, "family": fam, "netlist_id": f"{fam}_{seed0 + i}",
                        "variant": variant, "routed_fraction": prev, "n_components": len(bv.components),
                    })
                    if verbose:
                        print(f"[{done}/{total}] {sid}: skip (已存 rf={prev:.3f})", flush=True)
                    continue
                if verbose:
                    print(f"[{done}/{total}] {sid}: 重試 (前次失敗)", flush=True)
            label = label_fn(bv)
            rf = label["routed_fraction"] if isinstance(label, dict) else label
            g, r = board_to_graph(bv), board_to_raster(bv, size=raster_size)
            np.savez(npz_path, x=g["x"], edge_index=g["edge_index"], raster=r,
                     routed_fraction=np.float32(rf if rf is not None else -1.0))
            if verbose:
                print(f"[{done}/{total}] {sid}: rf={rf}", flush=True)
            manifest.append({
                "id": sid, "family": fam, "netlist_id": f"{fam}_{seed0 + i}",
                "variant": variant, "routed_fraction": rf, "n_components": len(bv.components),
            })
    _write_manifest(out_dir, manifest)
    return manifest


def _write_manifest(out_dir: str, rows: list[dict]) -> None:
    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def rebuild_manifest(out_dir: str) -> list[dict]:
    """從現有 .npz 重建 manifest.jsonl（run 被中斷、manifest 還沒寫時用；不重跑標註）。

    sid 格式 'fam_seed_variant'（fam/variant 皆無底線）→ 直接解析；
    routed_fraction、n_components 從 .npz 讀。
    """
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".npz"):
            continue
        parts = fn[:-4].split("_")
        if len(parts) != 3:
            continue
        fam, seed, variant = parts
        s = np.load(os.path.join(out_dir, fn))
        rf = float(s["routed_fraction"])
        rows.append({
            "id": fn[:-4], "family": fam, "netlist_id": f"{fam}_{seed}",
            "variant": variant, "routed_fraction": None if rf < 0 else rf,
            "n_components": int(s["x"].shape[0]),
        })
    _write_manifest(out_dir, rows)
    return rows
