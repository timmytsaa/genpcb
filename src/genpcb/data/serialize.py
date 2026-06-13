"""Compact placement DSL（輸出格式 v0；docs/output-format.md）。

設計動機（來自 tokenizer 煙霧測試）：現代 tokenizer single-digit splitting 使
浮點座標 `105.473` 要 ~6.6 tokens，原始 .kicad_pcb 小板就破 32k context。
對策：
1. **只表達 placement + netlist**（形態 A），不含 routing tracks 與 pad 幾何
   （後者由 footprint 型號隱含）。
2. **座標格點量化成整數**（預設 0.1mm 格），消滅小數點與多餘位數。
格式自我描述（header 帶 grid），可逆，dsl_to_board() round-trip 還原。
"""

from __future__ import annotations

from genpcb.data.procedural import Board, Component, Net


def board_to_dsl(board: Board, grid: float = 0.1) -> str:
    def q(v: float) -> int:
        return round(v / grid)

    lines = [f"B {q(board.w)} {q(board.h)} {board.layers} {grid}"]
    for c in board.components:
        lines.append(f"C {c.ref} {c.fp} {q(c.x)} {q(c.y)} {c.rot} {c.side}")
    for n in board.nets:
        pins = " ".join(f"{ref}.{pad}" for ref, pad in n.pins)
        lines.append(f"N {n.name} {pins}")
    return "\n".join(lines) + "\n"


def dsl_to_sft_example(text: str) -> dict[str, str]:
    """把 canonical DSL 切成 placement 任務的 (prompt, completion)。

    - prompt = 板框 B + 元件宣告 D（ref + footprint，無座標）+ netlist N + "PLACE"
    - completion = 擺位 P（ref x y rot side）

    這就是 GRPO 的 prompt 格式（同 netlist → 同 prompt → 同 group）。純字串轉換、
    不經 mm 量化，故對 canonical text 精確可逆（見 sft_example_to_dsl）。
    """
    blines, dlines, plines, nlines = [], [], [], []
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "B":
            blines.append(line)
        elif t[0] == "C":
            _, ref, fp, x, y, rot, side = t
            dlines.append(f"D {ref} {fp}")
            plines.append(f"P {ref} {x} {y} {rot} {side}")
        elif t[0] == "N":
            nlines.append(line)
    prompt = "\n".join(blines + dlines + nlines) + "\nPLACE\n"
    completion = "\n".join(plines) + "\n"
    return {"prompt": prompt, "completion": completion}


def sft_example_to_dsl(prompt: str, completion: str) -> str:
    """還原 dsl_to_sft_example：(prompt, completion) → canonical B/C/N DSL。

    GRPO 端解析 policy 輸出（completion）成 board 走這條：prompt 給元件宣告與
    netlist、completion 給擺位，合併成 canonical DSL 後即可 dsl_to_board()。
    """
    decls: dict[str, str] = {}
    blines, nlines = [], []
    for line in prompt.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "B":
            blines.append(line)
        elif t[0] == "D":
            decls[t[1]] = t[2]
        elif t[0] == "N":
            nlines.append(line)
    clines = []
    for line in completion.splitlines():
        t = line.split()
        if t and t[0] == "P":
            ref, x, y, rot, side = t[1], t[2], t[3], t[4], t[5]
            clines.append(f"C {ref} {decls[ref]} {x} {y} {rot} {side}")
    return "\n".join(blines + clines + nlines) + "\n"


def dsl_to_board(text: str) -> Board:
    comps: list[Component] = []
    nets: list[Net] = []
    w = h = 0.0
    layers, grid = 2, 0.1
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "B":
            wq, hq, layers, grid = int(t[1]), int(t[2]), int(t[3]), float(t[4])
            w, h = wq * grid, hq * grid
        elif t[0] == "C":
            _, ref, fp, xq, yq, rot, side = t
            comps.append(Component(ref, fp, int(xq) * grid, int(yq) * grid, int(rot), side))
        elif t[0] == "N":
            nets.append(Net(t[1], [tuple(p.split(".")) for p in t[2:]]))
    return Board(w, h, layers, comps, nets)
