"""每個 footprint 型號的 pad 幾何（近似）。

回傳 [(pad_num, x_off, y_off, pad_w, pad_h)]，x/y 為元件本地座標（mm），KiCad 在載入
時依 footprint 的旋轉角套用。幾何為「足夠佈線」的近似值，非 KiCad 官方庫精確尺寸——
精修待 KiCad 環境驗證（routability 只需 pad 數正確、位置合理、net 對得上）。

關鍵不變量：每個型號回傳恰好 npads 個、編號 "1".."npads" 的 pad（與 procedural 的
netlist 用同一套 pad 編號，故 net 對應自動一致）。
"""

from __future__ import annotations

from genpcb.data.procedural import FOOTPRINTS


def _single_row(n: int, pitch: float, pw: float, ph: float) -> list[tuple]:
    x0 = -(n - 1) / 2 * pitch
    return [(str(i + 1), x0 + i * pitch, 0.0, pw, ph) for i in range(n)]


def _dual_row(n: int, pitch: float, dx: float, pw: float, ph: float) -> list[tuple]:
    half = n // 2
    pads = []
    y0 = -(half - 1) / 2 * pitch
    for i in range(half):                       # 左排 1..half（上到下）
        pads.append((str(i + 1), -dx, y0 + i * pitch, pw, ph))
    for i in range(n - half):                   # 右排（下到上）
        pads.append((str(half + i + 1), dx, -(y0 + i * pitch), pw, ph))
    return pads


def _quad(n: int, w: float, h: float, pw: float = 0.3, ph: float = 0.8) -> list[tuple]:
    per = n // 4
    extra = n - per * 4
    sides = [per + (1 if s < extra else 0) for s in range(4)]
    pads, num = [], 1
    # 左(x=-w/2)↓、下(y=h/2)→、右(x=w/2)↑、上(y=-h/2)←，逆時針標準
    for side, cnt in enumerate(sides):
        for i in range(cnt):
            t = (i + 0.5) / cnt - 0.5            # -0.5..0.5
            if side == 0:
                x, y = -w / 2, t * h
            elif side == 1:
                x, y = t * w, h / 2
            elif side == 2:
                x, y = w / 2, -t * h
            else:
                x, y = -t * w, -h / 2
            pads.append((str(num), x, y, pw, ph))
            num += 1
    return pads


def pads_for(fp: str) -> list[tuple]:
    w, h, n = FOOTPRINTS[fp]
    if fp in ("C0402", "R0402", "IND1210"):
        return [("1", -w * 0.4, 0.0, w * 0.5, h), ("2", w * 0.4, 0.0, w * 0.5, h)]
    if fp == "SOT23":
        return [("1", -0.95, -1.0, 0.6, 0.7), ("2", 0.95, -1.0, 0.6, 0.7), ("3", 0.0, 1.0, 0.6, 0.7)]
    if fp == "XTAL":
        c = [(-w * 0.35, -h * 0.3), (w * 0.35, -h * 0.3), (w * 0.35, h * 0.3), (-w * 0.35, h * 0.3)]
        return [(str(i + 1), x, y, 1.0, 1.0) for i, (x, y) in enumerate(c)]
    if fp == "LDO223":                          # SOT-223：3 腳 + 散熱 tab
        return [("1", -2.3, 2.3, 0.8, 1.5), ("2", 0.0, 2.3, 0.8, 1.5),
                ("3", 2.3, 2.3, 0.8, 1.5), ("4", 0.0, -2.3, 3.0, 2.0)]
    if fp == "SOIC8":
        return _dual_row(n, pitch=1.27, dx=w * 0.45, pw=0.6, ph=1.2)
    if fp in ("QFN32", "QFP48"):
        return _quad(n, w, h)
    if fp == "CONN1x6":
        return _single_row(n, pitch=2.54, pw=1.2, ph=1.8)
    if fp == "CONN2x5":
        return _dual_row(n, pitch=2.54, dx=2.54 / 2, pw=1.0, ph=1.6)
    if fp == "USB_C":
        return _dual_row(n, pitch=0.8, dx=w * 0.4, pw=0.4, ph=1.2)
    return _single_row(n, pitch=0.5, pw=0.4, ph=0.4)   # fallback
