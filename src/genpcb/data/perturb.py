"""Stream X：擾動增強——從一個基準擺位產出同 netlist 的品質光譜。

用途：surrogate 訓練資料的「同 netlist 多品質擺位」群組（= ranking group，data-engine §1.3/§3）。
退化算子保留 netlist 不變、只動座標，故同群組 netlist 相同。
"""

from __future__ import annotations

import copy
import random

from genpcb.data.procedural import Board


def _clamp(board: Board) -> None:
    for c in board.components:
        w, h = c.size
        c.x = min(max(c.x, w / 2), max(w / 2, board.w - w / 2))
        c.y = min(max(c.y, h / 2), max(h / 2, board.h - h / 2))


def jitter(board: Board, sigma: float, seed: int = 0) -> Board:
    b = copy.deepcopy(board)
    rng = random.Random(seed)
    for c in b.components:
        c.x += rng.gauss(0, sigma)
        c.y += rng.gauss(0, sigma)
    _clamp(b)
    return b


def swap(board: Board, k: int, seed: int = 0) -> Board:
    b = copy.deepcopy(board)
    rng = random.Random(seed)
    n = len(b.components)
    for _ in range(k):
        i, j = rng.randrange(n), rng.randrange(n)
        b.components[i].x, b.components[j].x = b.components[j].x, b.components[i].x
        b.components[i].y, b.components[j].y = b.components[j].y, b.components[i].y
    return b


def scatter(board: Board, frac: float, seed: int = 0) -> Board:
    b = copy.deepcopy(board)
    rng = random.Random(seed)
    k = max(1, int(len(b.components) * frac))
    for c in rng.sample(b.components, k):
        w, h = c.size
        c.x = rng.uniform(w / 2, max(w / 2, b.w - w / 2))
        c.y = rng.uniform(h / 2, max(h / 2, b.h - h / 2))
    return b


def quality_spectrum(board: Board, seed: int = 0) -> list[tuple[str, Board]]:
    """基準 + 漸強擾動 → 同 netlist 的品質光譜（含標籤）。"""
    return [
        ("base", board),
        ("jitter1", jitter(board, 1.0, seed)),
        ("jitter5", jitter(board, 5.0, seed)),
        ("jitter20", jitter(board, 20.0, seed)),
        ("scatter25", scatter(board, 0.25, seed)),
        ("scatter50", scatter(board, 0.5, seed)),
        ("scatter100", scatter(board, 1.0, seed)),
        ("swap3", swap(board, 3, seed)),
    ]
