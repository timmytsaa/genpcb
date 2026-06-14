"""IR → Specctra DSN（Freerouting 的輸入格式）。

不依賴 pcbnew（避開 Colab 的 Python 綁定問題）；結構照 KiCad pcbnew 的 DSN 匯出慣例。
單位 um。Freerouting 佈線後輸出 .ses，由 parse_ses 抽 routed_fraction。
**Freerouting 實際接受度需在有 Java/Freerouting 的環境驗證**（本機僅驗結構）。
"""

from __future__ import annotations

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
         "  (resolution um 1)",          # 1 unit = 1 um；座標 = mm*1000（_um）
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
    """SES 的 net 級佈線估計（用 sexp parser，非字串比對）。

    注意：權威的連線級 routed_fraction 應取自 Freerouting log
    （route.parse_freerouting_log）；此處為 SES 幾何的輔助統計。
    """
    from genpcb.kicad.sexp import parse, walk
    try:
        tree = parse(ses_text)
    except Exception:
        return {"routed_fraction_ses": None, "n_wires": 0, "n_vias": 0}
    nets = walk(tree, "net")
    routed = sum(1 for nt in nets
                 if any(isinstance(c, list) and c and c[0] == "wire" for c in nt))
    n = len(nets) or 1
    return {
        "n_nets_out": len(nets),
        "n_routed_nets": routed,
        "routed_fraction_ses": routed / n,
        "n_wires": len(walk(tree, "wire")),
        "n_vias": len(walk(tree, "via")),
    }
