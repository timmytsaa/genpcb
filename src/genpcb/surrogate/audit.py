"""DAgger 式 audit：拿 GRPO rollout 的板，跑真 Freerouting 比對 surrogate 預測。

偵測 surrogate 漂移（policy 鑽進訓練分佈外），把真值回灌訓練集重訓 surrogate。
慢（每板數十秒），故只抽樣 k 個（通常 surrogate 評分最高的——policy 最愛去的地方）。
需 Freerouting 環境（本機或 reward farm）。
"""

from __future__ import annotations

import os

import numpy as np

from genpcb.data.procedural import Board
from genpcb.rewards.surrogate import surrogate_routability
from genpcb.surrogate.features import board_to_graph, board_to_raster


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def audit(boards: list[Board], model, jar: str = "tools/freerouting.jar",
          k: int = 8, strategy: str = "top", max_passes: int = 5) -> tuple[list[dict], dict]:
    """抽 k 個板跑真 Freerouting，比對 surrogate 預測。

    strategy='top'：抽 surrogate 評分最高的 k 個（policy 最愛去之處，最該查）。
    回傳 (rows[{pred,real,board}], drift{audit_spearman, audit_mae, n}）。
    """
    from genpcb.kicad.route import routing_reward

    preds = [surrogate_routability(b, model) for b in boards]
    order = sorted(range(len(boards)), key=lambda i: -preds[i])
    idx = order[:k] if strategy == "top" else order[:: max(1, len(order) // k)][:k]

    rows = []
    for i in idx:
        real = routing_reward(boards[i], jar=jar, max_passes=max_passes).get("routed_fraction")
        if real is not None:
            rows.append({"pred": preds[i], "real": real, "board": boards[i]})
    p = [r["pred"] for r in rows]
    q = [r["real"] for r in rows]
    drift = {
        "n": len(rows),
        "audit_spearman": _spearman(p, q),
        "audit_mae": float(np.mean(np.abs(np.array(p) - np.array(q)))) if rows else float("nan"),
    }
    return rows, drift


def append_to_dataset(rows: list[dict], data_dir: str, tag: str = "audit") -> int:
    """把 audited (board, real_rf) 存成 npz 回灌訓練集（DAgger 重訓用）。"""
    os.makedirs(data_dir, exist_ok=True)
    written = 0
    for j, r in enumerate(rows):
        b = r["board"]
        g, ra = board_to_graph(b), board_to_raster(b)
        # netlist_id 用 tag 分群，避免與既有 fam_seed_variant 撞名
        sid = f"{tag}_{abs(hash((tuple(c.ref for c in b.components), j))) % 10**8}_v"
        np.savez(os.path.join(data_dir, sid + ".npz"),
                 x=g["x"], edge_index=g["edge_index"], raster=ra,
                 routed_fraction=np.float32(r["real"]))
        written += 1
    return written
