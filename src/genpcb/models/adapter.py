"""模型家族差異的唯一棲身處。

sft.py / grpo.py / smoke_tokenizer.py 一律透過 ModelAdapter 取得
tokenizer / model / rollout 設定；本檔以外不得出現 if family 分支。
（A/B 紀律見 docs/generator-model-selection.md §3.5）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    family: str                # "qwen" | "gemma"
    model_id: str
    max_seq_len: int
    vllm_fast_inference: bool  # GRPO rollout 生成路徑


class ModelAdapter:
    def __init__(self, cfg: dict, model_id_override: str | None = None):
        m = cfg["model"]
        self.spec = ModelSpec(
            family=m["family"],
            model_id=model_id_override or m["model_id"],
            max_seq_len=int(m["max_seq_len"]),
            vllm_fast_inference=bool(m["vllm_fast_inference"]),
        )

    # ---- 不需要 GPU 的部分 --------------------------------------------------

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.spec.model_id)

    # ---- 訓練端（lazy import；煙霧測試不會走到這裡） --------------------------

    def load_for_training(self):
        """回傳 (model, tokenizer)。Unsloth 路徑，依 CUDA 環境另行安裝。"""
        from unsloth import FastLanguageModel

        return FastLanguageModel.from_pretrained(
            model_name=self.spec.model_id,
            max_seq_length=self.spec.max_seq_len,
            load_in_4bit=False,
        )

    def grpo_trainer_kwargs(self) -> dict:
        """GRPO rollout 生成路徑的家族差異。

        Gemma 4（2026-06 現況）：Unsloth 的 GRPO 尚不支援 vLLM fast
        inference，須退回 Unsloth 內建推理（吞吐較低）。此差異由 config 的
        vllm_fast_inference 表達，上游修復後改 config 即可，不改程式碼。
        """
        return {"use_vllm": self.spec.vllm_fast_inference}
