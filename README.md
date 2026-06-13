# genpcb

GRPO 訓練 LLM 生成 PCB；reward 採三層架構（確定性引擎 / 物理求解器 / surrogate）。

## 設計文件

- [docs/placement-routing-checker.md](docs/placement-routing-checker.md) — 檢查器（reward）設計：routing 全確定性、placement 訓 routability surrogate ensemble
- [docs/data-engine.md](docs/data-engine.md) — 訓練資料引擎：四條資料流 × 四條標註產線
- [docs/generator-model-selection.md](docs/generator-model-selection.md) — 生成器選型（Qwen3.5-9B 主力、Gemma 4 12B 挑戰者）與 A/B 工程形態
- [docs/output-format.md](docs/output-format.md) — 生成器輸出格式 v0：compact placement DSL（座標壓縮，~75× 小於原始 .kicad_pcb）

## 快速開始

```powershell
pip install -e .

# tokenizer 煙霧測試（不需 GPU；A/B 第一關）
python -m genpcb.train.smoke_tokenizer --config configs/model_qwen35_9b.yaml --config configs/model_gemma4_12b.yaml

# 產生 SFT 種子資料（不需 GPU；或用 notebooks/prepare_data.ipynb）
python -m genpcb.data.build --n 3000 --out data/sft/boards.jsonl --tokenizer google/gemma-4-12b-it

# 訓練（Colab，建議）：自成一本的 notebook，依序跑
#   notebooks/train_sft_colab.ipynb   QLoRA SFT（產資料→載入→LoRA→訓練→reward 檢查）
#   notebooks/train_grpo_colab.ipynb  GRPO（接續 SFT adapter，直接最佳化 placement reward）

# 訓練（本機 script，GPU）：另裝 train extras
pip install -e .[train]
python -m genpcb.train.sft  --config configs/model_gemma4_12b.yaml          # placement SFT，斷線可加 --resume
python -m genpcb.train.grpo --config configs/model_gemma4_12b.yaml          # 待 Phase 0 reward
```

## 結構

```
configs/            base.yaml + 每模型一份 config（差異最小化）
src/genpcb/
  models/adapter.py 模型家族差異的唯一棲身處
  train/            smoke_tokenizer（可跑）、sft、grpo（骨架）
  data/             資料引擎：procedural 生成 + DSL 序列化 + build（可跑）
  rewards/          compute_reward 聚合入口（Phase 0）
notebooks/          分析專用，只消費不生產
experiments/        run 輸出（config hash 命名）
```
