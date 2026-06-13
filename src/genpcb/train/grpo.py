"""GRPO 主訓練迴圈（骨架）。

Reward 介面唯一依賴 genpcb.rewards.compute_reward（分層聚合入口）；
rollout 生成路徑的家族差異由 ModelAdapter.grpo_trainer_kwargs() 表達。

用法：
    python -m genpcb.train.grpo --config configs/model_qwen35_9b.yaml
"""

from __future__ import annotations

import argparse

from genpcb.config import load_config
from genpcb.models.adapter import ModelAdapter
from genpcb.rewards import compute_reward  # noqa: F401  (Phase 0 介面)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-root", default=None, help="覆蓋 experiments 根目錄（Colab：Drive）")
    ap.add_argument("--resume", action="store_true", help="從最新 checkpoint 續訓")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.output_root:
        cfg["paths"]["experiments"] = args.output_root

    adapter = ModelAdapter(cfg)
    model, tokenizer = adapter.load_for_training()

    # TODO(Phase 3):
    #   1. prompt 集 = netlist/規格描述（同 prompt 群組 = 同 netlist，對齊 ranking 訓練）
    #   2. reward_fn: 解析 completion → 落地 .kicad_pcb → compute_reward()
    #      （解析失敗 = 大負 reward，不丟例外）
    #   3. trl.GRPOTrainer(..., num_generations=cfg["grpo"]["group_size"],
    #        **adapter.grpo_trainer_kwargs())
    #   4. 每 cfg["grpo"]["audit_every_steps"] 步：top-K rollout 丟真值產線、
    #      偵測 surrogate 漂移（placement-routing-checker.md §6）
    raise NotImplementedError("GRPO 迴圈待 Phase 0 reward 與 SFT checkpoint 就緒後實作")


if __name__ == "__main__":
    main()
