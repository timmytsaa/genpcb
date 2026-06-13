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
from genpcb.rewards import reward_from_completion


def make_reward_fn(weights: dict):
    """TRL GRPOTrainer 的 reward 函式：(prompts, completions) → list[float]。

    每個 completion 經 reward_from_completion 解析+評分；malformed/incomplete → floor。
    reward 函式本身純 Python、已本機測試（見 rewards/__init__.py）。
    """
    def reward_fn(prompts, completions, **_):
        return [reward_from_completion(p, c, weights) for p, c in zip(prompts, completions)]
    return reward_fn


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

    reward_fn = make_reward_fn(cfg["reward"])  # 已實作且本機測試（Tier-1 placement）

    # TODO(Phase 3，需 GPU）:
    #   1. prompt 集 = SFT 同款 placement prompt（同 netlist = 同 group，dsl_to_sft_example）
    #   2. trl.GRPOTrainer(model, reward_funcs=reward_fn,
    #        num_generations=cfg["grpo"]["group_size"], **adapter.grpo_trainer_kwargs())
    #   3. 每 cfg["grpo"]["audit_every_steps"] 步：top-K rollout 丟真值產線（Freerouting，
    #      需 KiCad 環境）、偵測 surrogate 漂移（placement-routing-checker.md §6）
    #   4. Tier-1 routing/DRC reward（dsl→.kicad_pcb）就緒後併入 reward_fn
    raise NotImplementedError(
        "GRPO 訓練迴圈（GPU）待接 TRL GRPOTrainer；reward_fn 已就緒。"
        "需 SFT checkpoint 暖身後啟動。"
    )


if __name__ == "__main__":
    main()
