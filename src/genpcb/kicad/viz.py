"""佈線視覺化（matplotlib，可在 Colab inline 顯示）。

plot_routing(board, ses_text)：板框 + 元件 pad + Freerouting 佈出的走線/via。
ses_text=None 則只畫 placement。這是「實際看到 PCB 佈線圖」最快的方式（不需 KiCad render）。
"""

from __future__ import annotations

import math

from genpcb.data.procedural import Board
from genpcb.kicad.footprints import pads_for
from genpcb.kicad.sexp import parse, walk

_LAYER_COLOR = {"F.Cu": "#c0392b", "B.Cu": "#2471a3"}


def parse_ses_geometry(ses_text: str, to_mm: float = 1000.0) -> dict:
    """從 SES 抽走線與 via 幾何（mm）。

    wire: (wire (path <layer> <width> x1 y1 x2 y2 ...)) → (layer, [(x,y)...])
    via:  (via <padstack> x y) → (x, y)
    座標單位同 DSN 的 resolution（um 1 → 值即 um，/1000 得 mm）。
    """
    tree = parse(ses_text)
    wires, vias = [], []
    for w in walk(tree, "wire"):
        path = next((c for c in w if isinstance(c, list) and c and c[0] == "path"), None)
        if path and len(path) >= 5:
            layer = path[1]
            nums = [float(v) for v in path[3:]]
            pts = [(nums[i] / to_mm, nums[i + 1] / to_mm) for i in range(0, len(nums) - 1, 2)]
            wires.append((layer, pts))
    for v in walk(tree, "via"):
        try:
            vias.append((float(v[-2]) / to_mm, float(v[-1]) / to_mm))
        except (ValueError, IndexError):
            pass
    return {"wires": wires, "vias": vias}


def _pad_rect(c, px, py, pw, ph):
    a = math.radians(c.rot)
    rx = px * math.cos(a) - py * math.sin(a)
    ry = px * math.sin(a) + py * math.cos(a)
    if c.rot in (90, 270):
        pw, ph = ph, pw
    return c.x + rx - pw / 2, c.y + ry - ph / 2, pw, ph


def plot_routing(board: Board, ses_text: str | None = None, ax=None, labels: bool = True):
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8 * max(0.4, board.h / max(board.w, 1e-6))))
    ax.add_patch(mp.Rectangle((0, 0), board.w, board.h, fill=False, ec="#27ae60", lw=1.5))

    for c in board.components:
        for num, px, py, pw, ph in pads_for(c.fp):
            x, y, w, h = _pad_rect(c, px, py, pw, ph)
            ax.add_patch(mp.Rectangle((x, y), w, h, color="#e67e22", alpha=0.6))
        if labels:
            ax.text(c.x, c.y, c.ref, fontsize=6, ha="center", va="center", zorder=5)

    routed = None
    if ses_text:
        g = parse_ses_geometry(ses_text)
        for layer, pts in g["wires"]:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=_LAYER_COLOR.get(layer, "#7f8c8d"), lw=0.9)
        for vx, vy in g["vias"]:
            ax.plot(vx, vy, "o", color="black", ms=2.5)
        routed = (len(g["wires"]), len(g["vias"]))

    ax.set_aspect("equal")
    ax.set_xlim(-2, board.w + 2)
    ax.set_ylim(board.h + 2, -2)              # PCB y 朝下
    ax.set_title(f"{len(board.components)} comp / {len(board.nets)} net"
                 + (f" / {routed[0]} wire {routed[1]} via" if routed else " (placement only)"))
    return ax
