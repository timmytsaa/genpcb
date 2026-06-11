"""分層 reward（docs/placement-routing-checker.md）。

Tier 1 確定性：pcb_metrics（KiCad DRC / ratsnest / HPWL / RUDY / 領域規則）
Tier 3 surrogate：Model A ensemble 推理 + uncertainty-gated audit
"""

from __future__ import annotations


def compute_reward(board_path: str) -> float:
    """Reward 聚合入口：R = w1·Tier1 + w2·ModelA − 硬規則懲罰。

    GRPO 迴圈唯一依賴的介面；內部組成可以換，簽名不動。
    """
    raise NotImplementedError(
        "Phase 0 未實作：pcb_metrics（KiCad DRC / ratsnest / HPWL / RUDY）。"
        "見 docs/placement-routing-checker.md §7 Phase 0。"
    )
