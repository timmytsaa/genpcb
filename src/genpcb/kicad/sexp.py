"""極簡 S-expression parser（KiCad .kicad_pcb / Specctra .dsn/.ses 共用）。"""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    toks, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "()":
            toks.append(ch)
            i += 1
        elif ch == '"':
            j, buf = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1]); j += 2
                else:
                    buf.append(text[j]); j += 1
            toks.append('"' + "".join(buf))      # 前綴 " 標記字串原子
            i = j + 1
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < n and text[j] not in "()\" \t\r\n":
                j += 1
            toks.append(text[i:j]); i = j
    return toks


def parse(text: str) -> list:
    toks = tokenize(text)
    pos = [0]

    def node():
        pos[0] += 1                                # 跳過 '('
        out = []
        while toks[pos[0]] != ")":
            if toks[pos[0]] == "(":
                out.append(node())
            else:
                t = toks[pos[0]]
                out.append(t[1:] if t.startswith('"') else t)
                pos[0] += 1
        pos[0] += 1                                # 跳過 ')'
        return out

    return node()


def head(node, name: str) -> list:
    """node 直接子節點中 head==name 者。"""
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def first(node, name: str):
    got = head(node, name)
    return got[0] if got else None


def walk(node, name: str):
    """遞迴找出所有 head==name 的子節點。"""
    out = []
    if isinstance(node, list):
        if node and node[0] == name:
            out.append(node)
        for c in node:
            if isinstance(c, list):
                out.extend(walk(c, name))
    return out
