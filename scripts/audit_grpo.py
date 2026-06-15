"""對 GRPO rollouts 跑真 Freerouting，比對 surrogate 預測 + 程序化基線。

決定性測試：GRPO 把 surrogate 分推高了，但**真實 routed_fraction 真的比基線高嗎**？
- GRPO real > baseline real → 真進步
- GRPO surrogate 高、但 GRPO real ≈ baseline 或更低 → surrogate hacking
"""

from __future__ import annotations

import json

import numpy as np

from genpcb.data.procedural import FAMILIES, generate_board
from genpcb.rewards import _board_from_completion
from genpcb.rewards.surrogate import load_surrogate, surrogate_routability
from genpcb.kicad.route import routing_reward

JAR = "tools/freerouting.jar"
m = load_surrogate("models/surrogate_a.pt")
rollouts = json.load(open("tools/grpo_rollouts.json"))

rows = []
for i, (prompt, gen) in enumerate(rollouts):
    try:
        gb = _board_from_completion(prompt, gen)
    except ValueError:
        print(f"{i}: GRPO completion 不合法（malformed/incomplete）", flush=True)
        continue
    bb = generate_board(FAMILIES[i % 3], seed=20000 + i)        # 同 netlist 的程序化基線
    pred = surrogate_routability(gb, m)
    g_real = routing_reward(gb, jar=JAR).get("routed_fraction")
    b_real = routing_reward(bb, jar=JAR).get("routed_fraction")
    rows.append({"i": i, "pred": pred, "grpo_real": g_real, "base_real": b_real})
    print(f"{i:2d}: GRPO surrogate={pred:.2f} real={g_real}  | baseline real={b_real}", flush=True)

ok = [r for r in rows if r["grpo_real"] is not None and r["base_real"] is not None]
gp = np.array([r["pred"] for r in ok])
gr = np.array([r["grpo_real"] for r in ok])
br = np.array([r["base_real"] for r in ok])
print("\n=== 決定性結果 ===")
print(f"合法 rollouts: {len(ok)}/{len(rollouts)}")
print(f"GRPO    真實 routed_fraction: mean {gr.mean():.3f}")
print(f"基線    真實 routed_fraction: mean {br.mean():.3f}")
print(f"GRPO 真實勝過基線比例: {(gr > br).mean():.0%}")
print(f"surrogate 預測 vs GRPO 真實 MAE: {np.mean(np.abs(gp - gr)):.3f}  (大 → surrogate 被鑽)")
print(f"surrogate 高估量 (pred - real) mean: {(gp - gr).mean():+.3f}  (正大 → hacking)")
