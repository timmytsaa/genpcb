# 生成器模型選型（2026-06 更新）

> 狀態：已決策（2026-06-11），取代先前「Qwen2.5-Coder-7B 起步、1.5B 快速迭代」的結論。
> 注意：本文件基於 2026-06 的網路資訊（次級來源），**實作前到 Hugging Face 確認 model card 與 base 權重存在**。

## 0. 先定決策規則，再定型號

開源模型每月在換代（Qwen3.5 之後 Qwen3.6 已出、3.7 在路上），鎖死型號沒有意義。鎖的是規則：

> **訓練端選型規則**：dense 架構、Apache 2.0、有 base checkpoint、Unsloth/verl 的 GRPO 路徑已驗證、≤10B（迭代速度）。實作開跑那天取「滿足全部條件的最新模型」。

以 2026-06 的資訊套用規則，結果如下。

## 1. 選型結果

| 角色 | 原決策 | 新決策 | 理由 |
|------|--------|--------|------|
| 訓練端主力 | Qwen2.5-Coder-7B（2024-09） | **Qwen3.5-9B（base）** | 2026-02 釋出，dense（RL 最穩）、Apache 2.0、單卡 GRPO 可行；base 程式能力遠超 Qwen2.5-Coder-7B |
| 快速迭代 | Qwen2.5-Coder-1.5B | **Qwen3.5-2B / 4B** | 同家族換小杯，管線零改動 |
| 規劃端（凍結） | Qwen3-Coder-Next | **不變** | 80B-A3B、SWE-bench Verified ~70.6%、46GB 單機可跑，仍是凍結規劃端的甜蜜點 |
| 進階擴模（Phase 3 後） | — | **Qwen3-Coder-30B-A3B 或 Qwen3.5-35B-A3B（MoE）** | Unsloth 已支援 MoE fine-tune/RL；等 dense 管線驗證過再上 |

## 2. 理由細節

1. **PCB 生成是窄域 DSL**（`.kicad_pcb` S-expr 或 pcbnew script），能力天花板主要由 SFT+RL 決定，base 模型挑「新且訓練順」比挑「程式榜最強」重要。榜上最強的 GLM-5.1 / Kimi K2.6 / DeepSeek V4 / MiniMax M3 全是超大 MoE，不是可訓練端的選項。
2. **Dense 優先於 MoE 做 GRPO**：MoE 在 RL 下的 router 不穩定是已知痛點，Unsloth 2026 的 MoE 支援讓它「可做」，但研究迭代期先用 dense 排除一個變因。
3. **Qwen3.5 小模型原生多模態（文字+影像）**——這開了一個新選項：把板面 raster（congestion/density render）當影像回饋餵給 policy，做多輪 placement 修正。**標記為 Phase 3+ 實驗**，第一版仍走純文字。
4. 換代成本低：同為 Qwen 系 tokenizer/chat template 親緣高，SFT/GRPO 管線（Unsloth 或 verl）對 Qwen3.5 已有現成路徑。

## 2.5 候選評估：Gemma 4 12B（2026-06-11 增補）

Gemma 4（2026-04-02 釋出：E2B / E4B / 26B-A4B MoE / 31B dense；**12B 於 2026-06-03 追加**）。用選型規則逐條評：

| 規則 | Gemma 4 12B | 評 |
|------|-------------|-----|
| Dense | ✔（26B 才是 MoE；12B 為 dense，待 HF 確認） | 過 |
| 授權 | ✔ **這代改 Apache 2.0**（Gemma 3 是自訂條款） | 過，重大改善 |
| Base checkpoint | ❓ 目前搜得到的是 -it 版，pt/base 待確認 | 驗證清單項 |
| GRPO 路徑 | ⚠ Unsloth 支援，但**目前 Gemma 4 的 GRPO 要關掉 vLLM fast inference 改用 Unsloth inference** | 扣分：GRPO 牆鐘被 rollout 生成主導，這直接拖慢整個 RL 迴圈 |
| ≤10B | 12B 略超 | 可接受（無預算限制） |

加分項：31B 在 Arena 文字榜開源第 3，家族底子強；12B 是 encoder-free 統一多模態（影像+音訊），對 Phase 3+「板面 raster 視覺回饋」實驗比 Qwen3.5 的架構更乾淨。

扣分項：除了 GRPO rollout 路徑不成熟，12B 是 6/3 才出的——一週新，tokenizer/chat template/推理引擎的生態 teething 期通常要幾週。

**決策：Qwen3.5-9B 維持主力，Gemma 4 12B 列為指定挑戰者。** 無預算限制下兩顆並行跑同一套 SFT 煙霧測試（tokenizer 對 S-expression 效率、SFT loss 曲線、GRPO rollout 吞吐），用數據定勝負，不用榜單定。若 Gemma 的 vLLM GRPO 路徑在開訓前修好，重新評估。

## 3. 不變的事

- 檢查器模型（Model A ensemble、B 家族）是 ~5M 級 GNN/UNet，與 LLM 換代無關。
- Reward 不用 LLM judge（既定決策，不因新模型改變）。

## 3.5 A/B 的工程形態（2026-06-11 增補）

**不用兩本 notebook。** A/B 的有效性取決於「除模型外一切相同」，兩份獨立 notebook 必然程式碼漂移，比較失效。定版形態：

```
genpcb/
├── configs/
│   ├── base.yaml                # 共同超參、資料路徑、reward 設定
│   ├── model_qwen35_9b.yaml     # 只放模型差異
│   └── model_gemma4_12b.yaml    # 只放模型差異
├── src/genpcb/
│   ├── models/adapter.py        # 唯一允許出現 if model_family 的地方：
│   │                            #   載入、chat template、Gemma GRPO 的
│   │                            #   vLLM fast inference 開關
│   ├── train/
│   │   ├── smoke_tokenizer.py   # S-expr token 效率實測（跑兩次，換 config）
│   │   ├── sft.py
│   │   └── grpo.py
│   ├── data/                    # 資料引擎（見 data-engine.md）
│   └── rewards/                 # pcb_metrics + surrogate 推理
├── notebooks/                   # 只做分析：讀 W&B/artifacts 畫比較圖
└── experiments/                 # run 輸出，以 config hash 命名
```

三條紀律：

1. **模型差異全部關進 `models/adapter.py`**——Gemma 4 GRPO 要關 vLLM fast inference 這種 quirk 是 adapter 裡的一個 flag，不是另一本 notebook 的理由。
2. **訓練一律用 script 跑**（CLI + config），不在 notebook 裡訓：GRPO 是數天級長跑，kernel 斷線 = run 報廢；script 才有 resume、才能丟 farm、run 名以 config hash 對應 W&B。
3. **Notebook 只消費 artifacts、不生產**：tokenizer 比較圖、SFT loss 對照、rollout 吞吐分析放 notebooks/，資料來源是 experiments/ 與 W&B，這樣 notebook 爛掉也不影響可重現性。

煙霧測試階段想用 notebook 當啟動器可以，但同一本、吃 config 參數（papermill 式），跑完即棄。

## 3.6 Colab 工作流（2026-06-11 增補）

訓練環境定為 Colab。紀律不變，只改啟動器形態：

- **唯一 launcher**：`notebooks/colab_launcher.ipynb`——兩個模型共用，改 `MODEL_CONFIG` 一個變數切換 A/B。cell 裡只有環境準備與 `!python -m genpcb.train.*`，不寫訓練邏輯。
- **程式碼進 Colab 的路**：repo push 到 GitHub（private + Colab Secrets 的 token），launcher `git clone/pull`。不要把 .py 散裝上傳 Drive——版本會失控。
- **Checkpoint 紀律**：Colab session 必死（搶占/timeout），checkpoint 一律落 Drive（`/content/drive/MyDrive/genpcb/experiments`）或 push HF Hub；sft/grpo script 要支援 `--resume`（Phase 2 實作項）。斷線恢復 = 重跑全部 cell，從最新 checkpoint 續。
- **GPU 現實**：A100 40GB → 9B/12B 走 **LoRA/QLoRA + Unsloth**（其 GRPO 路徑本來就是 Colab-first），full fine-tune 不可行。這與選 Unsloth 的決策互相印證。
- **分工**：Colab 只做 SFT/GRPO（GPU 活）。資料引擎——Freerouting/openEMS/FEniCSx 標註 farm——是 CPU 批次活，留在本機或其他機器跑，Colab 只吃已標註完的資料（從 Drive/HF datasets 讀）。
- 分析 notebook 留在本機，讀 W&B 與 Drive artifacts；launcher 不做分析。

## 3.7 Tokenizer 煙霧測試結果（2026-06-11 實測）

合成板語料 192k chars，實測（`experiments/smoke_tokenizer_*.json`）：

| tokenizer | vocab | chars/tok | tok/coord | tok/board |
|---|---|---|---|---|
| Qwen3.5-9B | 248k | 1.625 | 6.57 | 39,436 |
| Gemma 4 12B | 262k | 1.621 | 6.57 | 39,544 |
| Qwen2.5-Coder-1.5B | 152k | 1.664 | 6.57 | 38,526 |
| gpt2（對照） | 50k | 2.12 | 3.25 | 30,240 |

三個發現：

1. **兩個正式候選 ID 都真實存在且 tokenizer 可下載**（`Qwen/Qwen3.5-9B`、`google/gemma-4-12b-it`）——驗證清單前兩項部分打勾（weights 可用性仍需確認）。
2. **A/B 在 tokenizer 效率上是平手**（1.625 vs 1.621，tok/coord 完全相同）——這項不構成選型差異，勝負交給 SFT loss 與 GRPO 吞吐。
3. **真正的發現：現代 tokenizer 對座標數字有重稅。** 三個現代 tokenizer 都做 single-digit splitting（為算術能力），一個座標 `105.473` 要 6.57 tokens（gpt2 反而只要 3.25）。後果：40 元件 + 400 segment 的小板原始 S-expr 就要 ~39k tokens，**超過 32k context**。含佈線的完整板用原始格式直接出局；純 placement 輸出（形態 A）約 12–15k tokens 可行但浪費。→ **輸出格式定版必須包含座標壓縮**：格點量化成整數（如 0.05mm 格）、相對座標、或精簡 DSL。這是格式定版（開放問題 #1）的硬資料輸入。

> **已解（2026-06-11）**：開放問題 #1 形態 A 部分定版為 compact placement DSL（[output-format.md](output-format.md)）——只表達 placement+netlist、座標 0.1mm 格點量化成整數。實測 v0 種子板 Gemma tokenizer mean ~509、max ~1029 tokens，相對原始 .kicad_pcb 約 75× 壓縮。

## 4. 實作前驗證清單

- [ ] HF 上確認 `Qwen3.5-9B` base 權重與授權
- [ ] HF 上確認 Gemma 4 12B 是否有 pt/base 權重、是否 dense
- [ ] Unsloth/verl 跑通 Qwen3.5-9B 與 Gemma 4 12B 的 GRPO 煙霧測試（玩具 reward），記錄 rollout 吞吐
- [ ] Tokenizer 對 `.kicad_pcb` S-expression 的 token 效率實測：Qwen3.5 vs Gemma 4（256k vocab），與輸出格式定版連動
- [ ] 追蹤 Gemma 4 GRPO 的 vLLM fast inference 支援進度
- [ ] 若選 MoE 擴模：先查 router 穩定性的社群回報再投入
