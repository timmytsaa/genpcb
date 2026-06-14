"""IR → Specctra DSN（Freerouting 的輸入格式）。

不依賴 pcbnew（避開 Colab 的 Python 綁定問題）；結構照 KiCad pcbnew 的 DSN 匯出慣例。
單位 um。Freerouting 佈線後輸出 .ses，由 parse_ses 抽 routed_fraction。
**Freerouting 實際接受度需在有 Java/Freerouting 的環境驗證**（本機僅驗結構）。
"""

from __future__ import annotations

import re

from genpcb.data.procedural import Board
from genpcb.kicad.footprints import pads_for

_VIA = "Via[0-1]"


def _um(mm: float) -> int:
    return int(round(mm * 1000))


def board_to_dsn(board: Board, name: str = "genpcb", width_um: int = 250, clearance_um: int = 200) -> str:
    # padstacks（依 pad 尺寸去重）與 images（每 footprint 型號的 pin 佈局）
    padstacks: dict[tuple[int, int], str] = {}
    images: dict[str, list[tuple[str, str, int, int]]] = {}
    for fp in sorted({c.fp for c in board.components}):
        pins = []
        for num, x, y, pw, ph in pads_for(fp):
            wu, hu = _um(pw), _um(ph)
            pid = f"pad_{wu}x{hu}"
            padstacks[(wu, hu)] = pid
            pins.append((num, pid, _um(x), _um(y)))
        images[fp] = pins

    by_fp: dict[str, list] = {}
    for c in board.components:
        by_fp.setdefault(c.fp, []).append(c)

    L = [f"(pcb {name}.dsn",
         '  (parser (string_quote ") (space_in_quoted_tokens on) (host_cad "genpcb") (host_version "0.1"))',
         "  (resolution um 10)",
         "  (unit um)",
         "  (structure",
         "    (layer F.Cu (type signal) (property (index 0)))",
         "    (layer B.Cu (type signal) (property (index 1)))",
         f"    (boundary (rect pcb 0 0 {_um(board.w)} {_um(board.h)}))",
         f'    (via "{_VIA}")',
         f"    (rule (width {width_um}) (clearance {clearance_um}))",
         "  )",
         "  (placement"]
    for fp, comps in by_fp.items():
        L.append(f"    (component {fp}")
        for c in comps:
            side = "front" if c.side == "T" else "back"
            L.append(f"      (place {c.ref} {_um(c.x)} {_um(c.y)} {side} {c.rot})")
        L.append("    )")
    L.append("  )")
    L.append("  (library")
    for fp, pins in images.items():
        L.append(f"    (image {fp}")
        for num, pid, xu, yu in pins:
            L.append(f"      (pin {pid} {num} {xu} {yu})")
        L.append("    )")
    for (wu, hu), pid in padstacks.items():
        L += [f"    (padstack {pid}",
              f"      (shape (rect F.Cu {-wu // 2} {-hu // 2} {wu // 2} {hu // 2}))",
              f"      (shape (rect B.Cu {-wu // 2} {-hu // 2} {wu // 2} {hu // 2}))",
              "      (attach off)",
              "    )"]
    L += [f'    (padstack "{_VIA}"',
          "      (shape (circle F.Cu 600))",
          "      (shape (circle B.Cu 600))",
          "      (attach off)",
          "    )",
          "  )"]
    L.append("  (network")
    for net in board.nets:
        pins = " ".join(f"{ref}-{pad}" for ref, pad in net.pins)
        L += [f'    (net "{net.name}"', f"      (pins {pins})", "    )"]
    allnets = " ".join(f'"{n.name}"' for n in board.nets)
    L += [f"    (class default {allnets}",
          f'      (circuit (use_via "{_VIA}"))',
          f"      (rule (width {width_um}) (clearance {clearance_um}))",
          "    )",
          "  )",
          "  (wiring",
          "  )",
          ")"]
    return "\n".join(L) + "\n"


def parse_ses_routed_fraction(dsn_text: str, ses_text: str) -> dict:
    """比對 DSN 宣告的連線總數 vs SES 佈出的 wire/via，估 routed_fraction。

    SES 的 (wire (net "X") ...) 段落代表佈通的連線。粗估：有 wire 的 net 比例。
    精確 routed_fraction（unconnected ratsnest）需 KiCad 匯回後算；此為快速代理。
    """
    dsn_nets = set(re.findall(r'\(net "([^"]+)"', dsn_text))
    routed_nets = set(re.findall(r'\(net "([^"]+)"', ses_text))   # SES wire 段也用 (net "..")
    routed_nets &= dsn_nets
    n = len(dsn_nets) or 1
    return {
        "n_nets": len(dsn_nets),
        "n_routed_nets": len(routed_nets),
        "routed_fraction": len(routed_nets) / n,
        "n_wires": ses_text.count("(wire "),
        "n_vias": ses_text.count("(via "),
    }
