"""Tier-1 確定性 placement 指標（docs/placement-routing-checker.md §2.1）。

純幾何、操作於 Board IR，**不需要 KiCad**，每個 rollout 都可即時算。
pad 位置以元件中心近似（placement 階層的標準做法）。routing/DRC 指標（需
KiCad/Freerouting）屬另一層，見 rewards/__init__.py 的說明。
"""

from __future__ import annotations

from genpcb.data.procedural import Board, Component

# 需要去耦的 IC（decap proximity 規則的對象）
_POWER_IC = {"QFP48", "QFN32", "SOIC8"}


def _bbox(c: Component) -> tuple[float, float, float, float]:
    w, h = c.size
    return (c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2)


def _total_area(board: Board) -> float:
    return sum(c.size[0] * c.size[1] for c in board.components) or 1.0


def courtyard_overlap(board: Board) -> tuple[float, int]:
    """回傳 (重疊面積 / 元件總面積, 重疊對數)。硬違規。"""
    comps = board.components
    area, count = 0.0, 0
    for i in range(len(comps)):
        ax0, ay0, ax1, ay1 = _bbox(comps[i])
        for j in range(i + 1, len(comps)):
            bx0, by0, bx1, by1 = _bbox(comps[j])
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox > 0 and oy > 0:
                area += ox * oy
                count += 1
    return area / _total_area(board), count


def out_of_bounds(board: Board) -> tuple[float, int]:
    """回傳 (越界面積近似 / 元件總面積, 越界元件數)。硬違規。"""
    W, H = board.w, board.h
    area, count = 0.0, 0
    for c in board.components:
        x0, y0, x1, y1 = _bbox(c)
        ox = max(0.0, -x0) + max(0.0, x1 - W)
        oy = max(0.0, -y0) + max(0.0, y1 - H)
        if ox > 0 or oy > 0:
            count += 1
            w, h = c.size
            area += ox * h + oy * w
    return area / _total_area(board), count


def hpwl(board: Board) -> float:
    """半周長線長總和（net 成員元件中心的 bbox 半周長和）。"""
    ctr = {c.ref: (c.x, c.y) for c in board.components}
    total = 0.0
    for n in board.nets:
        xs = [ctr[r][0] for r, _ in n.pins if r in ctr]
        ys = [ctr[r][1] for r, _ in n.pins if r in ctr]
        if len(xs) >= 2:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def rudy(board: Board, bins: int = 32) -> tuple[float, float]:
    """RUDY congestion 估計，回傳 (峰值, 平均)。

    每個 net 把其線長密度（半周長 / bbox 面積）灑進其 bbox 覆蓋的格子。
    """
    W, H = board.w, board.h
    if W <= 0 or H <= 0:
        return 0.0, 0.0
    ctr = {c.ref: (c.x, c.y) for c in board.components}
    bw, bh = W / bins, H / bins
    grid = [0.0] * (bins * bins)
    for n in board.nets:
        xs = [ctr[r][0] for r, _ in n.pins if r in ctr]
        ys = [ctr[r][1] for r, _ in n.pins if r in ctr]
        if len(xs) < 2:
            continue
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        wnet = (x1 - x0) + (y1 - y0)
        area = max((x1 - x0) * (y1 - y0), bw * bh)   # 下限一格，避免除零
        dens = wnet / area
        i0, i1 = max(0, int(x0 // bw)), min(bins - 1, int(x1 // bw))
        j0, j1 = max(0, int(y0 // bh)), min(bins - 1, int(y1 // bh))
        for j in range(j0, j1 + 1):
            row = j * bins
            for i in range(i0, i1 + 1):
                grid[row + i] += dens
    return max(grid), sum(grid) / len(grid)


def decap_proximity(board: Board, threshold: float = 5.0) -> tuple[float, int]:
    """去耦電容離最近 IC 的超距懲罰（領域規則）。

    decap = 與某 IC 共網路的 C0402；回傳 (Σmax(0,dist−門檻) / 板對角線, decap 數)。
    """
    by_ref = {c.ref: c for c in board.components}
    ics = [c for c in board.components if c.fp in _POWER_IC]
    if not ics:
        return 0.0, 0
    ic_refs = {c.ref for c in ics}
    decaps: set[str] = set()
    for n in board.nets:
        refs = {r for r, _ in n.pins}
        if refs & ic_refs:
            decaps |= {r for r in refs if by_ref.get(r) and by_ref[r].fp == "C0402"}
    diag = (board.w ** 2 + board.h ** 2) ** 0.5 or 1.0
    pen = 0.0
    for r in decaps:
        c = by_ref[r]
        d = min(((c.x - ic.x) ** 2 + (c.y - ic.y) ** 2) ** 0.5 for ic in ics)
        pen += max(0.0, d - threshold)
    return pen / diag, len(decaps)


def placement_metrics(board: Board, rudy_bins: int = 32) -> dict:
    """一次算出全部 Tier-1 placement 指標。"""
    ov_area, ov_cnt = courtyard_overlap(board)
    oob_area, oob_cnt = out_of_bounds(board)
    rpeak, rmean = rudy(board, rudy_bins)
    dec, dec_cnt = decap_proximity(board)
    diag = (board.w ** 2 + board.h ** 2) ** 0.5 or 1.0
    hp = hpwl(board)
    return {
        "overlap_area": ov_area, "overlap_count": ov_cnt,
        "oob_area": oob_area, "oob_count": oob_cnt,
        "hpwl": hp,
        "hpwl_norm": hp / (diag * max(1, len(board.nets))),  # 平均每 net 跨幾條對角線
        "rudy_peak": rpeak, "rudy_mean": rmean,
        "decap_penalty": dec, "decap_count": dec_cnt,
    }
