"""過夜 orchestrator：循環 生成資料 → 重訓 surrogate → 記錄 grouped Spearman。

時間預算內持續擴大資料集，追蹤 grouped ρ 是否爬過 0.85（接 GRPO 的驗收門檻）。
無人值守、容錯（單輪失敗不中斷）。log → experiments/overnight_log.txt。

用法：python scripts/overnight_surrogate.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

JAR = "tools/freerouting.jar"
DATA = "data/surrogate_data"
LOG = "experiments/overnight_log.txt"
OUT = "experiments/surrogate_a.pt"
BUDGET_H = 9.0
TARGET = 0.85
CHUNK = 30          # 每輪新增 netlist 數
SEED_START = 40     # 接續目前批次（已做的 resume 會快速跳過）


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M')} {msg}"
    os.makedirs("experiments", exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def npz_count() -> int:
    return len([f for f in os.listdir(DATA) if f.endswith(".npz")]) if os.path.isdir(DATA) else 0


def main() -> None:
    from genpcb.kicad.route import routing_reward
    from genpcb.surrogate.gen_data import build_dataset

    label = lambda b: routing_reward(b, jar=JAR, max_passes=5)
    t0 = time.time()
    seed = SEED_START
    log(f"=== overnight start: budget {BUDGET_H}h, target grouped rho {TARGET}, chunk {CHUNK} ===")

    while time.time() - t0 < BUDGET_H * 3600:
        h = (time.time() - t0) / 3600
        try:
            log(f"[{h:.1f}h] generating netlists {seed}..{seed + CHUNK - 1}")
            build_dataset(CHUNK, DATA, seed0=seed, label_fn=label, verbose=False)
            seed += CHUNK
            r = subprocess.run(
                [sys.executable, "-m", "genpcb.surrogate.train", "--data", DATA,
                 "--epochs", "80", "--lam", "2.0", "--out", OUT],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            m = re.search(r"best grouped Spearman ([\d.]+)", r.stdout)
            rho = float(m.group(1)) if m else float("nan")
            log(f"[{(time.time()-t0)/3600:.1f}h] dataset {npz_count()} boards, grouped Spearman {rho:.3f}")
            if rho >= TARGET:
                log(f"*** TARGET REACHED: grouped Spearman {rho:.3f} >= {TARGET} ***")
                break
        except Exception as e:                    # 單輪失敗不中斷整夜
            log(f"[{h:.1f}h] iteration error: {type(e).__name__}: {e}")
            seed += CHUNK
    log(f"=== overnight done: {npz_count()} boards, {(time.time()-t0)/3600:.1f}h ===")


if __name__ == "__main__":
    main()
