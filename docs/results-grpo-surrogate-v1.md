# 結果記錄：GRPO v1（surrogate routability reward）+ 真值 audit

> 2026-06-15。第一次用 **routability surrogate 當 GRPO reward** 並以**真 Freerouting audit** 驗證。
> 結論：infrastructure 完整可信；但 LLM 擺位**尚未贏過程序化基線**，瓶頸是 SFT/LLM 品質、非 surrogate。

## 設定

- 起點：Gemma 4 12B QLoRA，SFT 暖身後 adapter
- reward：`make_surrogate_reward_fn`（surrogate routability 主導、overlap/越界硬 gate、HPWL 拿掉）
- GRPO：80 步、beta=0（KL 參考 base）、溫度 0.7、group=8
- audit：20 個 held-out rollout 跑真 Freerouting，與同 netlist 的程序化基線比

## 訓練（Colab）

- reward 從 **0.78 升到 2.6**（換算 surrogate_rf **0.16 → 0.57**），且 80 步時仍在升
- reward_std 1.55 → 0.74（收斂中）
- 合法率高（rollout 多為良構完整擺位）

## Audit（本機真 Freerouting，決定性測試）

| 指標 | 值 |
|------|-----|
| 合法 rollouts | 18/20 |
| **GRPO 真實 routed_fraction** | **0.608** |
| **程序化基線 真實** | **0.755** |
| GRPO 真實勝過基線比例 | **17%** |
| surrogate 高估量 (pred − real) | **−0.064** |
| surrogate pred vs GRPO real MAE | 0.148 |

## 三個結論

1. **Infrastructure 完美運作**：GRPO 最佳化 reward、audit 跑真值、**正確抓出「GRPO 沒贏基線」**。
   第一次能客觀量到 LLM 擺位的真實可佈線性——這套測量系統可信。
2. **不是 surrogate hacking**：高估量僅 −0.064（surrogate 對 GRPO 擺位的評分準）。
   排除「騙過 surrogate」這個假設。
3. **GRPO 真的有效但起點太爛**：surrogate_rf 0.16 → 0.57（80 步漲 3.5×、仍在升），
   但 **SFT 起點 0.16 太低 + 步數太少**，終點 0.57 未追上基線 0.755。

## 診斷：瓶頸是 SFT/LLM 擺位品質，不是 surrogate

- SFT 模型起點 surrogate_rf 僅 0.16 → **學會了格式、沒學會擺得好**（rollout 可見堆 cap 等退化擺位）。
- 程序化擺位（鋪得開）真實 0.755，是好範本——SFT 該學起來卻沒學好。
- surrogate（grouped ρ 0.78）對這些擺位評分準，**不是瓶頸**；DAgger 暫不需動。

## 下一步（優先序）

1. **強化 SFT（先做）**：讓 LLM 真的學會擺得好。
   - 用已生成的 2621 標註板，**挑 routed_fraction 高的擺位當 SFT 目標**（data-driven，比手工範本好）
   - 加 epoch/資料、確認 SFT 模型產出鋪得開的擺位（surrogate_rf 接近基線 ~0.75）
2. **GRPO 多跑**：80 步還在升 → 拉長到 200–500 步，看能否爬過 0.755。
3. **KL anchor 到 SFT**：beta=0 + 參考 base 容易漂到「堆 cap」局部解 → merge SFT 當 KL 參考
   （placement-routing-checker §6 待辦），防退化漂移。

## 可重現

- audit 腳本：`scripts/audit_grpo.py`（讀 rollouts json → GRPO vs 基線真值比對）
- surrogate 模型：`models/surrogate_a.pt`（grouped ρ ~0.77）
- 資料：`data/surrogate_data`（2621 板，gitignored）
