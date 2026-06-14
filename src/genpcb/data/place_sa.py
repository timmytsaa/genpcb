"""Simulated-annealing 擺位優化器（data-engine §1.1 的 sa_hpwl）。

產生「好」擺位 anchor：surrogate 品質光譜的高品質端、以及更好的 SFT 目標。
目標 = overlap + 越界 + HPWL + decap（與 Tier-1 placement reward 同，略過 RUDY 求快）。
"""

from __future__ import annotations

import copy
import math
import random

from genpcb.data.procedural import Board
from genpcb.rewards.metrics import courtyard_overlap, decap_proximity, hpwl, out_of_bounds

_W = {"overlap": 8.0, "bounds": 8.0, "hpwl": 1.0, "decap": 0.5}


def _cost(board: Board) -> float:
    ov, _ = courtyard_overlap(board)
    oob, oob_n = out_of_bounds(board)
    dec, _ = decap_proximity(board)
    diag = (board.w ** 2 + board.h ** 2) ** 0.5 or 1.0
    n = max(1, len(board.components))
    hp = hpwl(board) / (diag * max(1, len(board.nets)))
    return _W["overlap"] * ov + _W["bounds"] * (oob_n / n) + _W["hpwl"] * hp + _W["decap"] * dec


def sa_place(board: Board, steps: int = 3000, seed: int = 0,
             t0: float = 1.0, t1: float = 0.01) -> Board:
    """回傳優化後（cost 最低）的擺位副本。"""
    rng = random.Random(seed)
    b = copy.deepcopy(board)
    W, H = b.w, b.h
    n = len(b.components)
    cur = _cost(b)
    best, best_c = copy.deepcopy(b), cur
    for step in range(steps):
        T = t0 * (t1 / t0) ** (step / steps)
        c = b.components[rng.randrange(n)]
        ox, oy = c.x, c.y
        w, h = c.size
        if rng.random() < 0.75:                       # 局部抖動
            c.x += rng.gauss(0, max(1.0, W * 0.15))
            c.y += rng.gauss(0, max(1.0, H * 0.15))
        else:                                         # 全域重定位
            c.x, c.y = rng.uniform(w / 2, W - w / 2), rng.uniform(h / 2, H - h / 2)
        c.x = min(max(c.x, w / 2), max(w / 2, W - w / 2))
        c.y = min(max(c.y, h / 2), max(h / 2, H - h / 2))
        new = _cost(b)
        if new <= cur or rng.random() < math.exp((cur - new) / max(T, 1e-9)):
            cur = new
            if new < best_c:
                best, best_c = copy.deepcopy(b), new
        else:
            c.x, c.y = ox, oy                         # 還原
    return best
