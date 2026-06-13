"""Stream P：程序化板生成（docs/data-engine.md §1.1 的 v0）。

純 Python、無 KiCad 依賴，可在 Colab 跑。產出中性 IR（Board/Component/Net），
由 serialize.py 轉成生成器的目標 DSL。

v0 範圍與誠實聲明：
- 三個電路家族（mcu / buck / sensor），netlist 結構合法（每 net ≥2 pin），
  但**非電氣精確**——真正的 ERC-clean 模板文法是後續工作。
- placement 用簡單 rejection 擺放 + decap 靠近主 IC，是「合理 v0 種子」，
  非最佳化擺位；最佳化擺位需 §1.1 的 placement 合成器（SA/force-directed）。
- 用途：bootstrap SFT 種子資料、驗證 DSL token 長度。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# footprint -> (寬 mm, 高 mm, pad 數)
FOOTPRINTS: dict[str, tuple[float, float, int]] = {
    "QFP48": (9.0, 9.0, 48),
    "QFN32": (5.0, 5.0, 32),
    "SOIC8": (5.0, 4.0, 8),
    "SOT23": (3.0, 1.5, 3),
    "C0402": (1.0, 0.5, 2),
    "R0402": (1.0, 0.5, 2),
    "XTAL": (3.2, 2.5, 4),
    "LDO223": (6.5, 3.5, 4),
    "IND1210": (3.2, 2.5, 2),
    "USB_C": (9.0, 7.0, 16),
    "CONN1x6": (15.0, 2.5, 6),
    "CONN2x5": (13.0, 5.0, 10),
}

FAMILIES = ("mcu", "buck", "sensor")

_PASSIVE = {"C0402", "R0402", "IND1210"}


@dataclass
class Component:
    ref: str
    fp: str
    x: float = 0.0
    y: float = 0.0
    rot: int = 0       # 0/90/180/270
    side: str = "T"    # T/B

    @property
    def npads(self) -> int:
        return FOOTPRINTS[self.fp][2]

    @property
    def size(self) -> tuple[float, float]:
        w, h, _ = FOOTPRINTS[self.fp]
        return (h, w) if self.rot in (90, 270) else (w, h)


@dataclass
class Net:
    name: str
    pins: list[tuple[str, str]]  # (ref, pad)


@dataclass
class Board:
    w: float
    h: float
    layers: int
    components: list[Component] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)


# --- 元件清單（每家族） -----------------------------------------------------

def _make_components(family: str, rng: random.Random) -> list[Component]:
    comps: list[Component] = []
    counter: dict[str, int] = {}

    def add(fp: str, prefix: str) -> None:
        counter[prefix] = counter.get(prefix, 0) + 1
        comps.append(Component(f"{prefix}{counter[prefix]}", fp))

    if family == "mcu":
        add("QFP48" if rng.random() < 0.5 else "QFN32", "U")
        for _ in range(comps[0].npads // 8 + rng.randint(1, 3)):
            add("C0402", "C")            # decoupling 群
        add("XTAL", "Y"); add("C0402", "C"); add("C0402", "C")   # 晶振負載電容
        add("LDO223", "U"); add("C0402", "C"); add("C0402", "C")  # LDO in/out
        add("CONN1x6", "J")
        if rng.random() < 0.5:
            add("USB_C", "J")
    elif family == "buck":
        add("SOIC8", "U")                # controller
        add("IND1210", "L")
        for _ in range(rng.randint(2, 4)):
            add("C0402", "C")
        add("R0402", "R"); add("R0402", "R")   # 回授分壓
        add("CONN1x6", "J"); add("CONN1x6", "J")  # in / out
    elif family == "sensor":
        add("QFN32" if rng.random() < 0.5 else "SOT23", "U")
        for _ in range(rng.randint(1, 3)):
            add("C0402", "C")
        add("R0402", "R"); add("R0402", "R")   # I2C pull-up
        add("CONN1x6", "J")
    else:
        raise ValueError(f"unknown family: {family}")
    return comps


# --- netlist（結構合法：每 net ≥2 pin） -------------------------------------

def _make_nets(comps: list[Component], family: str, rng: random.Random) -> list[Net]:
    avail = {c.ref: [str(i) for i in range(1, c.npads + 1)] for c in comps}
    vrail = "VOUT" if family == "buck" else "+3V3"
    gnd = Net("GND", [])
    vcc = Net(vrail, [])

    for c in comps:  # 去耦電容：pad1→電源軌、pad2→GND
        if c.fp == "C0402" and len(avail[c.ref]) >= 2:
            vcc.pins.append((c.ref, avail[c.ref].pop(0)))
            gnd.pins.append((c.ref, avail[c.ref].pop(0)))

    for c in comps:  # 其餘元件：一腳 GND、（多數）一腳電源
        if c.fp == "C0402":
            continue
        if avail[c.ref]:
            gnd.pins.append((c.ref, avail[c.ref].pop(0)))
        if c.fp not in ("R0402", "XTAL", "IND1210") and avail[c.ref]:
            vcc.pins.append((c.ref, avail[c.ref].pop(0)))

    nets = [n for n in (gnd, vcc) if len(n.pins) >= 2]

    # 剩餘 pad 配成訊號 net：每 net 連「不同元件」的腳（避免短接同一 IC 自身的腳）
    remaining = {ref: list(ps) for ref, ps in avail.items() if ps}
    i = 0
    while sum(1 for ps in remaining.values() if ps) >= 2:   # 至少兩個元件還有腳可連
        refs = [r for r, ps in remaining.items() if ps]
        rng.shuffle(refs)
        k = 3 if (len(refs) >= 3 and rng.random() < 0.2) else 2
        chosen = refs[:k]
        nets.append(Net(f"N{i:03d}", [(r, remaining[r].pop()) for r in chosen]))
        for r in chosen:
            if not remaining[r]:
                del remaining[r]
        i += 1
    return nets   # 全屬同一元件的剩餘 pad 留作 NC


# --- placement（簡單 rejection 擺放） ---------------------------------------

def _bbox(c: Component) -> tuple[float, float, float, float]:
    w, h = c.size
    return (c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2)


def _overlap(a, b, margin: float) -> bool:
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0]
                or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def _board_size(comps: list[Component], density: float = 2.5) -> tuple[float, float]:
    area = sum(FOOTPRINTS[c.fp][0] * FOOTPRINTS[c.fp][1] for c in comps) * density
    side = max(20.0, area ** 0.5)
    return round(side, 1), round(side * 0.7, 1)


def _place(board: Board, rng: random.Random, margin: float = 0.5) -> None:
    placed: list[Component] = []
    for c in sorted(board.components, key=lambda c: -FOOTPRINTS[c.fp][2]):  # 大件先擺
        w, h = c.size
        for _ in range(200):
            c.x = rng.uniform(w / 2 + 1, max(w / 2 + 1.1, board.w - w / 2 - 1))
            c.y = rng.uniform(h / 2 + 1, max(h / 2 + 1.1, board.h - h / 2 - 1))
            if all(not _overlap(_bbox(c), _bbox(p), margin) for p in placed):
                break
        placed.append(c)


def generate_board(family: str = "mcu", seed: int = 0, layers: int = 2) -> Board:
    rng = random.Random(seed)
    comps = _make_components(family, rng)
    for c in comps:
        if c.fp in _PASSIVE and rng.random() < 0.5:
            c.rot = 90
    w, h = _board_size(comps)
    board = Board(w, h, layers, comps, [])
    _place(board, rng)
    board.nets = _make_nets(comps, family, rng)
    return board
