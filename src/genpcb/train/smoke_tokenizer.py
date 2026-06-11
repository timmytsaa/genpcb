"""Tokenizer 煙霧測試：實測候選模型對 .kicad_pcb S-expression 的 token 效率。

不需要 GPU。對每個候選 tokenizer 回報：
- chars/token：整體壓縮率（越高越好）
- tokens/coord：單一座標數字平均花費 token 數（此格式的成本主要在座標）
- tokens/board：合成樣板整板 token 數（估 context 預算用）

用法：
    python -m genpcb.train.smoke_tokenizer --config configs/model_qwen35_9b.yaml --config configs/model_gemma4_12b.yaml
    python -m genpcb.train.smoke_tokenizer --model-id Qwen/Qwen2.5-Coder-1.5B          # 直接指定 ID
    python -m genpcb.train.smoke_tokenizer --model-id gpt2 --boards "data/**/*.kicad_pcb"  # 用真實板取代合成板
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

from genpcb.config import load_config
from genpcb.models.adapter import ModelAdapter

HEADER = """(kicad_pcb (version 20240108) (generator "pcbnew")
  (general (thickness 1.6))
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (44 "Edge.Cuts" user)
  )
"""

FOOTPRINT_TMPL = """  (footprint "Capacitor_SMD:C_0402_1005Metric"
    (layer "F.Cu")
    (at {x:.3f} {y:.3f} {rot})
    (property "Reference" "C{ref}" (at 0 -1.43 {rot}) (layer "F.SilkS")
      (effects (font (size 1 1) (thickness 0.15))))
    (pad "1" smd roundrect (at -0.48 0 {rot}) (size 0.56 0.62)
      (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net_a} "{name_a}"))
    (pad "2" smd roundrect (at 0.48 0 {rot}) (size 0.56 0.62)
      (layers "F.Cu" "F.Paste" "F.Mask") (roundrect_rratio 0.25) (net {net_b} "{name_b}"))
  )
"""

SEGMENT_TMPL = """  (segment (start {x1:.3f} {y1:.3f}) (end {x2:.3f} {y2:.3f}) (width 0.25) (layer "{layer}") (net {net}))
"""

VIA_TMPL = """  (via (at {x:.3f} {y:.3f}) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net {net}))
"""


def synth_board(seed: int = 17, n_footprints: int = 40, n_segments: int = 400, n_vias: int = 60) -> str:
    """合成一張代表性小板的 S-expression（座標分佈與真實板同量級）。"""
    rng = random.Random(seed)
    nets = ['(net 0 "")', '(net 1 "GND")', '(net 2 "+3V3")'] + [
        f'(net {i} "Net-(U1-Pad{i})")' for i in range(3, 40)
    ]
    parts = [HEADER, "  " + "\n  ".join(nets) + "\n"]
    net_names = {1: "GND", 2: "+3V3"}
    for i in range(n_footprints):
        a, b = rng.randint(1, 39), rng.randint(1, 39)
        parts.append(FOOTPRINT_TMPL.format(
            x=rng.uniform(80, 180), y=rng.uniform(40, 120), rot=rng.choice([0, 90, 180, 270]),
            ref=i + 1, net_a=a, name_a=net_names.get(a, f"Net-(U1-Pad{a})"),
            net_b=b, name_b=net_names.get(b, f"Net-(U1-Pad{b})"),
        ))
    for _ in range(n_segments):
        x, y = rng.uniform(80, 180), rng.uniform(40, 120)
        parts.append(SEGMENT_TMPL.format(
            x1=x, y1=y, x2=x + rng.uniform(-10, 10), y2=y + rng.uniform(-10, 10),
            layer=rng.choice(["F.Cu", "B.Cu"]), net=rng.randint(1, 39),
        ))
    for _ in range(n_vias):
        parts.append(VIA_TMPL.format(x=rng.uniform(80, 180), y=rng.uniform(40, 120), net=rng.randint(1, 39)))
    parts.append(")\n")
    return "".join(parts)


def coord_samples(seed: int = 17, n: int = 200) -> list[str]:
    rng = random.Random(seed)
    return [f"{rng.uniform(0, 250):.3f}" for _ in range(n)]


def measure(tokenizer, boards: list[str], coords: list[str]) -> dict:
    total_chars = sum(len(b) for b in boards)
    t0 = time.perf_counter()
    board_tokens = [len(tokenizer.encode(b, add_special_tokens=False)) for b in boards]
    elapsed = time.perf_counter() - t0
    coord_tokens = [len(tokenizer.encode(c, add_special_tokens=False)) for c in coords]
    total_tokens = sum(board_tokens)
    return {
        "vocab_size": int(tokenizer.vocab_size),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "chars_per_token": round(total_chars / total_tokens, 3),
        "tokens_per_coord": round(sum(coord_tokens) / len(coord_tokens), 2),
        "tokens_per_board": round(total_tokens / len(boards)),
        "encode_seconds": round(elapsed, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", action="append", default=[], help="model config（可重複）")
    ap.add_argument("--model-id", action="append", default=[], help="直接指定 HF model id（可重複）")
    ap.add_argument("--boards", default=None, help=".kicad_pcb glob；不給則用合成板")
    ap.add_argument("--out", default=None, help="JSON 輸出路徑（預設 experiments/smoke_tokenizer_<ts>.json）")
    args = ap.parse_args()

    candidates: list[str] = []
    for cfg_path in args.config:
        candidates.append(ModelAdapter(load_config(cfg_path)).spec.model_id)
    candidates += args.model_id
    if not candidates:
        ap.error("至少給一個 --config 或 --model-id")

    if args.boards:
        paths = glob.glob(args.boards, recursive=True)
        boards = [Path(p).read_text(encoding="utf-8", errors="replace") for p in paths]
        source = f"{len(boards)} real boards ({args.boards})"
        if not boards:
            ap.error(f"glob 無匹配：{args.boards}")
    else:
        boards = [synth_board(seed=s) for s in (17, 18, 19)]
        source = "3 synthetic boards"
    coords = coord_samples()

    from transformers import AutoTokenizer

    results: dict[str, dict] = {}
    for model_id in candidates:
        print(f"[smoke] loading tokenizer: {model_id}")
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as exc:  # gated / 404 / 網路 — 記錄後繼續比其他顆
            print(f"[smoke]   SKIP: {type(exc).__name__}: {exc}")
            results[model_id] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        results[model_id] = measure(tok, boards, coords)

    width = max(len(m) for m in results)
    print(f"\ncorpus: {source}, {sum(len(b) for b in boards):,} chars")
    print(f"{'model':<{width}}  {'vocab':>8}  {'chars/tok':>9}  {'tok/coord':>9}  {'tok/board':>9}")
    for model_id, r in results.items():
        if "error" in r:
            print(f"{model_id:<{width}}  (skipped: {r['error'][:60]})")
        else:
            print(f"{model_id:<{width}}  {r['vocab_size']:>8,}  {r['chars_per_token']:>9}  "
                  f"{r['tokens_per_coord']:>9}  {r['tokens_per_board']:>9,}")

    out = Path(args.out) if args.out else Path("experiments") / f"smoke_tokenizer_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": source, "results": results}, indent=2), encoding="utf-8")
    print(f"\n[smoke] written: {out}")


if __name__ == "__main__":
    main()
