"""用已標註資料建「best-of-N」SFT 目標：每個 netlist 挑 routed_fraction 最高的擺位。

SFT v0 用任意程序化擺位當目標（routed~0.70），模型學到格式但擺位品質普通。
改用**已知佈通率最高的變體**當目標——免額外佈線（重用 surrogate 資料的 label），
讓 LLM 模仿「好擺位」而非平均擺位。
"""

from __future__ import annotations

import json
import os

import numpy as np

from genpcb.data.perturb import quality_spectrum
from genpcb.data.procedural import FAMILIES, generate_board
from genpcb.data.serialize import board_to_dsl


def build_best_sft(data_dir: str, out_jsonl: str) -> list[dict]:
    """掃 surrogate 資料 → 每 netlist 挑最高 routed_fraction 的變體 → 重建擺位 → 寫 SFT jsonl。"""
    best: dict[str, tuple] = {}                          # netlist_id -> (rf, fam, seed, variant)
    for fn in os.listdir(data_dir):
        if not fn.endswith(".npz"):
            continue
        parts = fn[:-4].split("_")
        if len(parts) != 3 or parts[0] not in FAMILIES:  # 跳過 audit_* 等
            continue
        fam, seed, variant = parts
        rf = float(np.load(os.path.join(data_dir, fn))["routed_fraction"])
        if rf < 0:
            continue
        nid = f"{fam}_{seed}"
        if nid not in best or rf > best[nid][0]:
            best[nid] = (rf, fam, int(seed), variant)

    rows = []
    for nid, (rf, fam, seed, variant) in sorted(best.items()):
        spec = dict(quality_spectrum(generate_board(fam, seed), seed=seed))   # 與生成時同 seed → 可重現
        board = spec[variant]
        rows.append({"netlist_id": nid, "variant": variant, "routed_fraction": rf,
                     "n_components": len(board.components), "text": board_to_dsl(board)})

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows
