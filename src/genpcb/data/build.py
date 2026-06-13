"""產出 SFT 種子資料集（D-sft v0）：程序化板 → compact DSL → jsonl。

每行一筆：{family, seed, n_components, n_nets, chars, text}。

用法：
    python -m genpcb.data.build --n 300 --out data/sft/boards.jsonl
    python -m genpcb.data.build --n 60  --tokenizer google/gemma-4-12b-it   # 附 token 長度統計
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genpcb.data.procedural import FAMILIES, generate_board
from genpcb.data.serialize import board_to_dsl, dsl_to_board


def build(n: int, out: str | Path, grid: float = 0.1, seed0: int = 0) -> list[dict]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out.open("w", encoding="utf-8") as f:
        for i in range(n):
            family = FAMILIES[i % len(FAMILIES)]
            board = generate_board(family, seed=seed0 + i)
            dsl = board_to_dsl(board, grid=grid)
            rt = dsl_to_board(dsl)  # round-trip 自檢
            assert len(rt.components) == len(board.components), "DSL round-trip 元件數不符"
            assert len(rt.nets) == len(board.nets), "DSL round-trip net 數不符"
            row = {
                "family": family,
                "seed": seed0 + i,
                "n_components": len(board.components),
                "n_nets": len(board.nets),
                "chars": len(dsl),
                "text": dsl,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    return rows


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/sft/boards.jsonl")
    ap.add_argument("--grid", type=float, default=0.1, help="座標格點 mm（越大越省 token）")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tokenizer", default=None, help="HF model id；給了就附 token 長度統計")
    args = ap.parse_args()

    rows = build(args.n, args.out, grid=args.grid, seed0=args.seed0)
    chars = [r["chars"] for r in rows]
    print(f"[build] {len(rows)} boards → {args.out}")
    print(f"[build] chars/board: mean {sum(chars) // len(chars):,}  "
          f"p50 {_percentile(chars, 50):,.0f}  p95 {_percentile(chars, 95):,.0f}  max {max(chars):,}")

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        toks = [len(tok.encode(r["text"], add_special_tokens=False)) for r in rows]
        over = sum(t > 32768 for t in toks)
        print(f"[build] tokenizer={args.tokenizer}")
        print(f"[build] tokens/board: mean {sum(toks) // len(toks):,}  "
              f"p50 {_percentile(toks, 50):,.0f}  p95 {_percentile(toks, 95):,.0f}  max {max(toks):,}")
        print(f"[build] 超過 32k context: {over}/{len(toks)}")


if __name__ == "__main__":
    main()
