"""IR → .kicad_pcb S-expression。

自成一檔（footprint pad 幾何內嵌），不依賴安裝 KiCad 庫；net 對應由 board.nets 決定。
側別 T/B → 元件 layer F/B.Cu。pad 暫統一 F 層（2 層板第一版，精修待 KiCad 驗證）。
"""

from __future__ import annotations

from genpcb.data.procedural import Board
from genpcb.kicad.footprints import pads_for

_HEADER = '''(kicad_pcb (version 20240108) (generator "genpcb")
  (general (thickness 1.6))
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (36 "B.SilkS" user)
    (37 "F.SilkS" user)
    (44 "Edge.Cuts" user)
  )
'''


def board_to_kicad_pcb(board: Board) -> str:
    # net id 指派：0 保留給未連線
    name_to_id = {"": 0}
    decl = ['  (net 0 "")']
    for i, net in enumerate(board.nets, start=1):
        name_to_id[net.name] = i
        decl.append(f'  (net {i} "{net.name}")')
    pin_net: dict[tuple[str, str], tuple[int, str]] = {}
    for net in board.nets:
        for ref, pad in net.pins:
            pin_net[(ref, pad)] = (name_to_id[net.name], net.name)

    parts = [_HEADER, "\n".join(decl), "\n"]
    for c in board.components:
        layer = "F.Cu" if c.side == "T" else "B.Cu"
        parts.append(f'  (footprint "genpcb:{c.fp}" (layer "{layer}") (at {c.x:.4f} {c.y:.4f} {c.rot})\n')
        parts.append(f'    (property "Reference" "{c.ref}" (at 0 0 {c.rot}) (layer "F.SilkS"))\n')
        for num, px, py, pw, ph in pads_for(c.fp):
            nid, nname = pin_net.get((c.ref, num), (0, ""))
            parts.append(
                f'    (pad "{num}" smd roundrect (at {px:.4f} {py:.4f} {c.rot}) (size {pw:.3f} {ph:.3f}) '
                f'(layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {nid} "{nname}"))\n'
            )
        parts.append('  )\n')

    parts.append(
        f'  (gr_rect (start 0 0) (end {board.w:.4f} {board.h:.4f}) '
        f'(stroke (width 0.1) (type solid)) (layer "Edge.Cuts"))\n'
    )
    parts.append(")\n")
    return "".join(parts)
