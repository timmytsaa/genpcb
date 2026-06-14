"""Board IR → surrogate 特徵（graph + raster）。純 numpy，本機可測。

- board_to_graph：節點=元件、邊=net clique；給 GNN 分支。
- board_to_raster：多通道板面影像（佔據/pad 密度/RUDY/mask）；給 UNet 分支。
congestion 本質是空間量 → raster；連通性 → graph。兩者融合（§3）。
"""

from __future__ import annotations

import math

import numpy as np

from genpcb.data.procedural import FOOTPRINTS, Board
from genpcb.kicad.footprints import pads_for

# 節點特徵維度： x, y, sin θ, cos θ, w, h, npads, is_IC, side
NODE_DIM = 9
RASTER_CHANNELS = ("occupancy", "pad_density", "rudy", "board_mask")


def board_to_graph(board: Board) -> dict:
    idx = {c.ref: i for i, c in enumerate(board.components)}
    feats = []
    for c in board.components:
        w, h, n = FOOTPRINTS[c.fp]
        feats.append([
            c.x / max(board.w, 1e-6), c.y / max(board.h, 1e-6),
            math.sin(math.radians(c.rot)), math.cos(math.radians(c.rot)),
            w / 10.0, h / 10.0, n / 48.0,
            1.0 if n >= 8 else 0.0,                 # is_IC
            0.0 if c.side == "T" else 1.0,          # side
        ])
    edges: set[tuple[int, int]] = set()
    for net in board.nets:
        members = sorted({idx[r] for r, _ in net.pins if r in idx})
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                edges.add((members[i], members[j]))
    edge_index = (np.array(sorted(edges), dtype=np.int64).T
                  if edges else np.zeros((2, 0), dtype=np.int64))
    return {
        "x": np.array(feats, dtype=np.float32).reshape(-1, NODE_DIM),
        "edge_index": edge_index,                    # [2, E]，無向（單向存）
        "refs": [c.ref for c in board.components],
    }


def board_to_raster(board: Board, size: int = 64) -> np.ndarray:
    H = W = size
    sx, sy = W / max(board.w, 1e-6), H / max(board.h, 1e-6)
    occ = np.zeros((H, W), np.float32)
    pad = np.zeros((H, W), np.float32)
    rudy = np.zeros((H, W), np.float32)
    mask = np.ones((H, W), np.float32)

    for c in board.components:
        w, h, _ = FOOTPRINTS[c.fp]
        if c.rot in (90, 270):
            w, h = h, w
        x0, x1 = int((c.x - w / 2) * sx), int((c.x + w / 2) * sx)
        y0, y1 = int((c.y - h / 2) * sy), int((c.y + h / 2) * sy)
        occ[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 1.0
        for _, px, py, _, _ in pads_for(c.fp):
            a = math.radians(c.rot)
            gx = int((c.x + px * math.cos(a) - py * math.sin(a)) * sx)
            gy = int((c.y + px * math.sin(a) + py * math.cos(a)) * sy)
            if 0 <= gx < W and 0 <= gy < H:
                pad[gy, gx] += 1.0

    cell_area = (board.w / W) * (board.h / H)
    ctr = {c.ref: (c.x, c.y) for c in board.components}
    for net in board.nets:
        xs = [ctr[r][0] for r, _ in net.pins if r in ctr]
        ys = [ctr[r][1] for r, _ in net.pins if r in ctr]
        if len(xs) < 2:
            continue
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        dens = ((x1 - x0) + (y1 - y0)) / max((x1 - x0) * (y1 - y0), cell_area)
        gx0, gx1 = max(0, int(x0 * sx)), min(W - 1, int(x1 * sx))
        gy0, gy1 = max(0, int(y0 * sy)), min(H - 1, int(y1 * sy))
        rudy[gy0:gy1 + 1, gx0:gx1 + 1] += dens

    if pad.max() > 0:
        pad /= pad.max()
    if rudy.max() > 0:
        rudy /= rudy.max()
    return np.stack([occ, pad, rudy, mask], axis=0)   # [C, H, W]
