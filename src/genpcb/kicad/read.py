""".kicad_pcb → IR。

通用 S-expression parser + 抽取 footprint/pad/net/板框。對本套 writer 的輸出精確
round-trip；對任意真實 KiCad 檔為 best-effort（footprint 型號回復假設名稱為
"genpcb:<TYPE>" 或最後一段；真實庫名→型號的對應表是 file-import 階段的後續工作）。
"""

from __future__ import annotations

from genpcb.data.procedural import Board, Component, Net
from genpcb.kicad.sexp import first as _first
from genpcb.kicad.sexp import head as _head
from genpcb.kicad.sexp import parse as _parse


def kicad_pcb_to_board(text: str) -> Board:
    tree = _parse(text)
    assert tree and tree[0] == "kicad_pcb", "不是 kicad_pcb"

    layers = _first(tree, "layers")
    n_cu = sum(1 for c in (layers or []) if isinstance(c, list) and len(c) >= 3 and c[2] == "signal")

    w = h = 0.0
    rect = _first(tree, "gr_rect")
    if rect:
        end = _first(rect, "end")
        if end:
            w, h = float(end[1]), float(end[2])

    comps: list[Component] = []
    pin_net: dict[str, list[tuple[str, str]]] = {}   # netname -> [(ref, pad)]
    for fp in _head(tree, "footprint"):
        fp_name = fp[1].split(":")[-1]
        at = _first(fp, "at")
        x, y = float(at[1]), float(at[2])
        rot = int(float(at[3])) if len(at) > 3 else 0
        layer = _first(fp, "layer")
        side = "B" if layer and str(layer[1]).startswith("B.") else "T"
        ref = "?"
        for prop in _head(fp, "property"):
            if prop[1] == "Reference":
                ref = prop[2]
        comps.append(Component(ref, fp_name, x, y, rot, side))
        for pad in _head(fp, "pad"):
            padnum = pad[1]
            net = _first(pad, "net")
            if net and len(net) >= 3 and net[2] != "":
                pin_net.setdefault(net[2], []).append((ref, padnum))

    nets = [Net(name, pins) for name, pins in pin_net.items()]
    return Board(w, h, n_cu or 2, comps, nets)
