"""SFT warm-start（骨架）。

資料來源：D-sft（docs/data-engine.md §5）——真實板 + 高品質合成板序列化後的
jsonl。在輸出格式定版（開放問題 #1）與資料引擎產出前，本 script 會在資料
檢查處明確停下。

用法：
    python -m genpcb.train.sft --config configs/model_qwen35_9b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from genpcb.config import load_config
from genpcb.models.adapter import ModelAdapter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    dataset_path = Path(cfg["sft"]["dataset"])
    if not dataset_path.exists():
        sys.exit(
            f"[sft] 資料集不存在：{dataset_path}\n"
            "先跑資料引擎產出 D-sft（docs/data-engine.md §5），"
            "且輸出格式需先定版（generator-model-selection.md §4 驗證清單）。"
        )

    adapter = ModelAdapter(cfg)
    model, tokenizer = adapter.load_for_training()

    # TODO(Phase 2):
    #   1. datasets.load_dataset("json", data_files=...) → 按 netlist 切分（防洩漏，data-engine.md §3）
    #   2. trl.SFTTrainer(model=model, processing_class=tokenizer,
    #        args=SFTConfig(num_train_epochs=cfg["sft"]["epochs"],
    #                       learning_rate=cfg["sft"]["lr"], ...))
    #   3. run 名 = config hash，輸出到 cfg["paths"]["experiments"]
    raise NotImplementedError("SFT 訓練迴圈待輸出格式定版後實作")


if __name__ == "__main__":
    main()
