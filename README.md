# genpcb

GRPO 訓練 LLM 生成 PCB；reward 採三層架構（確定性引擎 / 物理求解器 / surrogate）。

## 設計文件

- [docs/placement-routing-checker.md](docs/placement-routing-checker.md) — 檢查器（reward）設計：routing 全確定性、placement 訓 routability surrogate ensemble
- [docs/data-engine.md](docs/data-engine.md) — 訓練資料引擎：四條資料流 × 四條標註產線
- [docs/generator-model-selection.md](docs/generator-model-selection.md) — 生成器選型（Qwen3.5-9B 主力、Gemma 4 12B 挑戰者）與 A/B 工程形態

## 快速開始

```powershell
pip install -e .

# tokenizer 煙霧測試（不需 GPU；A/B 第一關）
python -m genpcb.train.smoke_tokenizer --config configs/model_qwen35_9b.yaml --config configs/model_gemma4_12b.yaml

# 訓練端（GPU，另裝 train extras 與 unsloth）
pip install -e .[train]
python -m genpcb.train.sft  --config configs/model_qwen35_9b.yaml
python -m genpcb.train.grpo --config configs/model_qwen35_9b.yaml
```

## 結構

```
configs/            base.yaml + 每模型一份 config（差異最小化）
src/genpcb/
  models/adapter.py 模型家族差異的唯一棲身處
  train/            smoke_tokenizer（可跑）、sft、grpo（骨架）
  data/             資料引擎（Phase 1）
  rewards/          compute_reward 聚合入口（Phase 0）
notebooks/          分析專用，只消費不生產
experiments/        run 輸出（config hash 命名）
```
