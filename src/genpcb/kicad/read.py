""".kicad_pcb → IR。

通用 S-expression parser + 抽取 footprint/pad/net/板框。對本套 writer 的輸出精確
round-trip；對任意真實 KiCad 檔為 best-effort（footprint 型號回復假設名稱為
"genpcb:<TYPE>" 或最後一段；真實庫名→型號的對應表是 file-import 階段的後續工作）。
"""

from __future__ import annotations

from genpcb.data.procedural import Board, Component, Net


def _tokenize(text: str) -> list[str]:
    toks, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "()":
            toks.append(ch)
            i += 1
        elif ch == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                else:
                    buf.append(text[j])
                    j += 1
            toks.append('"' + "".join(buf))      # 前綴 " 標記字串原子
            i = j + 1
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < n and text[j] not in "()\" \t\r\n":
                j += 1
            toks.append(text[i:j])
            i = j
    return toks


def _parse(toks: list[str]) -> list:
    pos = [0]

    def node():
        assert toks[pos[0]] == "("
        pos[0] += 1
        out = []
        while toks[pos[0]] != ")":
            out.append(node() if toks[pos[0]] == "(" else _atom(toks[pos[0]]))
            if toks[pos[0]] != ")" and not isinstance(out[-1], list):
                pos[0] += 1
        pos[0] += 1
        return out

    return node()


def _atom(tok: str):
    return tok[1:] if tok.startswith('"') else tok   # 去掉字串標記前綴


def _head(node, name):
    """node 內第一層、head 為 name 的子節點。"""
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def _first(node, name):
    got = _head(node, name)
    return got[0] if got else None


def kicad_pcb_to_board(text: str) -> Board:
    tree = _parse(_tokenize(text))
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
