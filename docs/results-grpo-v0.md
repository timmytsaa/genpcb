# 結果記錄：GRPO v0（placement proxy）

> 2026-06-14。第一個跑通的 GRPO 迴圈（Gemma 4 12B QLoRA，接續 SFT adapter，
> placement 確定性 reward）。記錄結論與從中學到的事。

## 經過

- 第一次嘗試（beta=0.001、二元 floor reward、溫度 1.0）：200 步後 **greedy 輸出 0% 合法**，
  格式從 SFT 退化。根因：二元 floor 在「整組都壞」時無梯度（冷啟動死穴）+ PEFT 的 KL 參考
  是 base、beta>0 把 policy 拉回沒學過格式的 base + 高溫取樣多不合法。
- 修法（commit cfe5baf）：**shaped reward**（malformed 給部分分數，恆有梯度往合法爬）
  + **beta=0** + **溫度 0.7**。
- 第二次（80 步）：**合法率 10/10 = 100%**，格式穩定不崩。

## 數據（held-out netlist，n=10）

| 指標 | 值 |
|------|-----|
| 合法率（parse + 完整） | **100%** |
| GRPO reward mean | -1.81 |
| 程序化基線 reward mean | -1.55 |
| 勝過基線比例 | 30% |

## 結論

1. **GRPO 迴圈驗證通過**：穩定、100% 合法、不崩。基礎設施成立。
2. **GRPO 沒有改善擺位品質**（≈基線，略低，差距在 n=10 雜訊內）。
   **原因 = placement proxy 飽和**：產生 SFT 資料的程序化基線本身已接近這個確定性 proxy
   的最佳值，proxy 上面沒有 headroom 可爬（訓練時 reward 曲線一路平在 -2 已是徵兆）。

## 學到的事（寫進設計原則）

- **GRPO 的價值來自「訓練資料沒有飽和」的 reward。** 在被基線飽和的 proxy 上，GRPO 無事可做。
- **冷啟動需要 reward 在「全錯」時也有梯度**（shaped，非二元 floor）。
- **PEFT 下 KL 參考是 base、不是 SFT**：beta>0 會把 policy 拉回 base → 格式崩。
  目前 beta=0 規避；正解是 merge SFT 當參考（待辦）。
- **bnb 4-bit 推論很慢（~3.7 tok/s）**：eval/生成要 merge 回 bf16 才快（也順帶得到正確 KL 參考）。

## 下一步（已轉向）

placement proxy 沒有 headroom → 轉去做**有 headroom 的 routability reward**（程序化擺位完全
沒為佈線最佳化過，那裡 GRPO 有真東西可學）。需 **P0：.kicad_pcb ↔ IR 橋**（純 Python 已完成
並 round-trip 驗證，見 genpcb/kicad/）+ Freerouting/DRC（需 KiCad 環境）。
