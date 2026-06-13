"""GRPO 主訓練迴圈（骨架）。

Reward 介面唯一依賴 genpcb.rewards.compute_reward（分層聚合入口）；
rollout 生成路徑的家族差異由 ModelAdapter.grpo_trainer_kwargs() 表達。

用法：
    python -m genpcb.train.grpo --config configs/model_qwen35_9b.yaml
"""

from __future__ import annotations

import argparse

from genpcb.config import load_config
from genpcb.data.procedural import FAMILIES, generate_board
from genpcb.data.serialize import board_to_dsl, dsl_to_sft_example
from genpcb.models.adapter import ModelAdapter
from genpcb.rewards import reward_from_completion


def make_reward_fn(weights: dict, parse_fail: float = -100.0):
    """TRL GRPOTrainer 的 reward 函式：(prompts, completions) → list[float]。

    每個 completion 經 reward_from_completion 解析+評分；malformed/incomplete → floor。
    reward 函式本身純 Python、已本機測試（見 rewards/__init__.py）。
    """
    def reward_fn(prompts, completions, **_):
        return [reward_from_completion(p, c, weights, parse_fail) for p, c in zip(prompts, completions)]
    return reward_fn


def build_prompts(n: int, seed0: int = 10000, grid: float = 0.1) -> list[dict]:
    """程序化產 GRPO prompt 集（只要 prompt，completion 由 policy 自生）。

    seed0 預設 10000，與 SFT 種子（0..）錯開，使 GRPO 在新 netlist 上最佳化、
    不只重現 SFT 看過的板。回傳 [{"prompt": ...}]。
    """
    out = []
    for i in range(n):
        fam = FAMILIES[i % len(FAMILIES)]
        ex = dsl_to_sft_example(board_to_dsl(generate_board(fam, seed=seed0 + i), grid=grid))
        out.append({"prompt": ex["prompt"]})
    return out


def _latest_checkpoint(out_dir: Path) -> str | None:
    cks = list(out_dir.glob("checkpoint-*"))
    return str(max(cks, key=lambda p: int(p.name.split("-")[1]))) if cks else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sft-adapter", required=True, help="SFT 暖身後的 LoRA adapter 目錄（GRPO 由此接續）")
    ap.add_argument("--output-root", default=None, help="覆蓋 experiments 根目錄（Colab：Drive）")
    ap.add_argument("--resume", action="store_true", help="從最新 checkpoint 續訓")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.output_root:
        cfg["paths"]["experiments"] = args.output_root
    g = cfg["grpo"]

    prompts = build_prompts(g["n_prompts"])           # 純 Python、本機可測
    reward_fn = make_reward_fn(cfg["reward"], parse_fail=g["parse_fail"])

    # ── GPU/訓練端；重依賴鎖在此 ──
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    spec = ModelAdapter(cfg).spec
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(spec.model_id, quantization_config=bnb,
                                                device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)  # 接續 SFT adapter
    model.config.use_cache = False

    out_dir = Path(cfg["paths"]["experiments"]) / f"grpo-{spec.family}"
    grpo_args = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=float(g["lr"]),
        beta=float(g["beta"]),
        num_generations=g["group_size"],
        per_device_train_batch_size=g["group_size"],
        gradient_accumulation_steps=2,
        max_prompt_length=g["max_prompt_length"],
        max_completion_length=g["max_completion_length"],
        max_steps=g["max_steps"],
        logging_steps=5,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=spec.vllm_fast_inference,
        report_to="none",
    )
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=grpo_args,
                          train_dataset=Dataset.from_list(prompts), processing_class=tok)
    resume = _latest_checkpoint(out_dir) if args.resume else None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir / "final"))
    print(f"[grpo] done → {out_dir / 'final'}")


if __name__ == "__main__":
    main()
