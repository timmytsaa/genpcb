# Placement / Routing 檢查器：設計決策與訓練計畫

> 狀態：已決策（2026-06-11）；同日增補 §9「無預算限制版」——模型擴成 ensemble + 專科家族，§0 的單模型版仍是理解架構的基準。
> 上游脈絡：GRPO 訓練 LLM 生成 PCB；reward 分三層（Tier 1 確定性 / Tier 2 物理求解器 / Tier 3 代理模型）。

---

## 0. 決策摘要

| # | 決策 | 一句話理由 |
|---|------|-----------|
| 1 | **Routing 檢查不訓練模型**（第一階段） | 安全關鍵的 routing 指標 100% 可從幾何確定性計算，毫秒級，訓練模型只會引入誤差 |
| 2 | **Placement 檢查訓練一顆 routability surrogate（Model A）**，這是第一階段唯一要訓練的檢查模型 | 「這個擺位好不好」的真值 = 自動佈線結果，但 autorouter 單板要 10–60 秒，進不了 GRPO 迴圈 |
| 3 | **ground truth 用 Freerouting 批次自動佈線產生**，固定 effort budget | autorouter 本身就是 routability 的 Tier 2「求解器」，不用等物理求解器 |
| 4 | 訓練資料 = **程序化 netlist 生成 + GitHub 真實 KiCad 板 + 擾動增強** | 程序化給覆蓋度，真實板給分佈真實性，擾動給廉價的品質光譜 |
| 5 | Loss 以 **同 netlist 群組內的 pairwise ranking** 為主 | GRPO 的 advantage 是 group 內相對排名，surrogate 只要排序對就夠，不必校準絕對值 |
| 6 | **反 reward-hacking：定期抽查 + surrogate 線上更新**（DAgger 式） | 凍結的 surrogate 在 RL 中必被攻破，這是已知鐵律 |
| 7 | 物理 surrogate（SI/串擾/IR drop，Model B）**延後到 Phase 4** | 先用 closed-form 與確定性檢查撐住，等 Model A 管線驗證過再複用同一套資料引擎 |

兩種 policy 形態都支援：

- **形態 A（建議起步）**：LLM 只做 placement，routing 交給 Freerouting。Model A 是 placement 的主 reward；最終驗收用真實佈線結果。
- **形態 B（後期）**：LLM 同時做 placement + routing。routing 部分用確定性檢查（§1），Model A 退居 placement 階段的 shaping reward。

Curriculum：先 A 後 B。

---

## 1. Routing 檢查：全部確定性，不訓練

所有指標從 `.kicad_pcb` 幾何直接算，工具為 `pcbnew` Python API + `kicad-cli` + `shapely`。

| 檢查項 | 計算方式 | Reward 形態 |
|--------|----------|------------|
| DRC 違規 | `kicad-cli pcb drc --format json`（KiCad 8+） | 違規數負懲罰（capped hinge）；間距/線寬餘裕當連續項 |
| 未連線網路 | `pcbnew` ratsnest unconnected count | 重懲罰（接近硬 gate） |
| 總線長 / via 數 / 層使用 | 遍歷 tracks/vias，純算術 | 連續項（與 HPWL 比值正規化） |
| Diff pair 間距 / 長度匹配 | 按 net class 抓 pair，逐段計算 gap 與 skew | 超出容差的量當負 reward |
| Bus 長度匹配 | 同 net class 內 max-min 長度差 | 同上 |
| 阻抗 | IPC-2141 closed-form（microstrip/stripline） | 偏離目標的百分比 |
| **Reference plane crossing**（回流路徑破壞） | 把訊號 track 投影到相鄰平面層，與平面 polygon 的 split/挖空做 `shapely` 交集，數跨越次數 | 每次跨越固定懲罰——這是 EMI 最大宗來源，幾何就能抓 |
| 關鍵網路 loop area | 訊號路徑與回流路徑圍出的多邊形面積 | 連續負項 |
| 可製造性（acid trap、銳角、孤銅） | 幾何角度/連通性檢查 | 違規數 |

**Tier 1.5（看情況進迴圈）**：IR drop——銅箔 polygon 上的 2D Laplace（sheet resistance）求解，小板粗網格用 FEniCSx 約 0.1–1 秒。策略：不是每個 rollout 都算，只對 group 內 Tier-1 得分前 k 名算，或每 N step 算一次。

> 結論：routing 的「檢查模型」就是一個確定性 Python library（`pcb_metrics`），不是神經網路。學習模型只在 §6 的 Model B（串擾/SI）才出現，且延後。

---

## 2. Placement 檢查 = Routability 預測問題

### 2.1 確定性部分（Tier 1，每 rollout 必算）

- Courtyard 重疊、板框越界、keepout 違規（DRC 子集）
- HPWL（總和 + per-net）
- **RUDY congestion map**（Rectangular Uniform wire DensitY）：峰值與平均
- Pin density map、net 直線交叉數
- 領域規則：decoupling cap 到 IC 電源腳距離、晶振到 MCU 距離、發熱元件間距、connector 板邊約束

### 2.2 學習部分（Model A）：要回答的問題

> 「給定這個 placement，自動佈線器在固定預算內能佈通多少？佈完品質如何？」

這在 IC EDA 是成熟問題（RouteNet、CircuitNet 的 congestion/DRC hotspot 預測；Google 2021 Nature graph placement 也是用 GNN proxy reward 預測 wirelength/congestion），但 PCB 尺度小得多（數十~數百元件），可以做得更輕。

**標籤定義（關鍵：固定 effort budget 才是良定義的標籤）**：
Freerouting 以固定參數（pass 數上限 + 60s timeout）佈線後記錄——

- `routed_fraction`：佈通網路比例
- `wl_ratio`：實際總線長 / HPWL
- `via_count`
- `per_net_routed`：每網路成功與否（dense 監督）
- `congestion_map`：佈線後銅密度 / 繞行比的 2D 圖（dense 監督）
- 匯回 KiCad 後的 DRC 違規數

---

## 3. Model A 架構規格

雙分支融合，總參數量控制在 ~5M，推理 <10ms（單張 GPU）：

```
板圖 (graph)                          板面 (raster, 256×256 多通道)
nodes = 元件/pad                      ch: 元件佔據、pin density、
  feat: 尺寸、pin數、(x,y,θ)、層、      RUDY、RUDY(power)、RUDY(signal)、
        類型(IC/被動/連接器)、locked     板框/keepout mask
edges = nets (hyperedge→star/clique)
  feat: net class、fanout
        │                                   │
   GNN 3–5 層                           小型 UNet
  (GraphSAGE/EdgeConv)                      │
        │                                   │
        └────────── 融合 (cross-attn 或 concat+MLP) ──────────┐
                                                              │
  輸出頭：                                                     │
  ① per-net routability (sigmoid)      ← dense，樣本效率主力   │
  ② congestion heatmap (UNet decoder)  ← dense                │
  ③ 純量組：routed_fraction / wl_ratio / via / DRC 數 ←────────┘
```

選 GNN 與使用者既有 gAAGNet/AAG 思路一脈相承；raster 分支是因為 congestion 本質是空間量，純 graph 抓不住。

---

## 4. 資料引擎

> 2026-06-11 起詳細設計移至 [data-engine.md](data-engine.md)（四條資料流 × 四條標註產線、placement 合成器、切分紀律、成本估算）。本節保留原始摘要。

### 4.1 板來源（三路並進）

1. **程序化生成**：參數化電路家族模板（MCU 板、buck converter、op-amp 濾波、connector breakout、小型 FPGA fanout），元件從 KiCad 官方庫取樣。對應 CAD 那套「程序化正向生成」方法論。
2. **真實板收集**：爬 GitHub 上開源 `.kicad_pcb`（注意授權過濾），拆掉 tracks 留 placement + netlist。
3. **擾動增強**：對好的 placement 做退化——位置 jitter（σ ∈ {1, 5, 20} mm）、旋轉隨機化、元件對調、cluster 打散。一塊板生 ~10 個品質光譜變體，**同 netlist 變體天然構成 ranking 訓練的群組**。

### 4.2 標註管線

```
.kicad_pcb (placement only)
  → pcbnew.ExportSpecctraDSN()          # 匯出 DSN
  → Freerouting headless (固定 budget)   # Java, 可平行多 JVM / Docker
  → .ses 匯回 → 抽取 §2.2 標籤
  → (graph, raster, labels) 存成訓練樣本
```

**規模估算**：10k 板 × 10 變體 = 100k 樣本；平均 30s/板 ≈ 35 CPU-day，16 工平行約 2–3 天跑完。可行。

---

## 5. 訓練與驗收

**Multi-task loss**：

```
L = BCE(per-net routability)            # dense 主力
  + MSE(congestion map)
  + MSE(純量組)
  + λ · PairwiseMarginRanking(同 netlist 群組內, 依 routed_fraction 排序)
```

**Ranking loss 是對齊 GRPO 的關鍵**：GRPO 的 advantage 是同 prompt（= 同 netlist）group 內正規化的相對值，所以 surrogate 在「同 netlist 不同 placement」之間排序正確 ≫ 絕對值校準。訓練資料的擾動變體群組正好就是這個分佈。

**驗收門檻（過了才准進 RL 迴圈）**：

- held-out 真實板、同 netlist 群組內 Spearman ρ ≥ 0.85（surrogate 分數 vs 實際 routed_fraction）
- per-net routability AUC ≥ 0.90
- 推理延遲 < 10ms/board

---

## 6. RL 迴圈整合與反 reward-hacking

**Reward 聚合**（形態 A，placement policy）：

```
R = w₁·Tier1_placement(確定性) + w₂·ModelA(routability surrogate) − 懲罰項(硬規則)
```

權重先手調，等指標分佈穩定後可上 CRITIC/AHP 客觀權重。早期讓確定性項佔主導，surrogate 權重隨其 audit 表現逐步上調。

**反作弊（不可省略）**：

1. 每 N 個 GRPO step，抽 group 內 surrogate 評分最高的 K 個 rollout → 跑真 Freerouting → 比對。
2. 比對結果**回灌 surrogate 訓練集、線上微調**（DAgger 式 active learning）——policy 分佈會漂移，凍結的 surrogate 必被攻破。
3. 監控 audit 樣本上的 Spearman；跌破 0.7 → 暫停 RL、補資料重訓 surrogate。

**Curriculum**：≤20 元件小板起步 → 漸增元件數與層數；形態 A 收斂後再開形態 B（policy 自己佈線，routing 確定性檢查接管主 reward）。

---

## 7. 階段排程

| Phase | 內容 | 產出 |
|-------|------|------|
| 0（~2 週） | `pcb_metrics` 確定性檢查庫：KiCad 9 + pcbnew + kicad-cli 包裝、graph/raster 抽取器 | Tier 1 reward 可用 |
| 1（2–3 週） | 資料引擎：netlist 生成器、GitHub 收集、擾動、Freerouting 批次標註 | 100k 標註樣本 |
| 2（2–3 週，與 1 部分重疊） | 訓練 Model A、過驗收門檻 | routability surrogate v1 |
| 3 | 接 GRPO：R = Tier1 + Model A，audit loop 上線 | placement RL 跑起來 |
| 4 | Model B 物理 surrogate（openEMS 串擾、FEniCSx IR drop），複用 Phase 1 資料引擎 | SI/PI 進迴圈 |

---

## 8. 技術棧定版

- **KiCad 9.x**：`kicad-cli pcb drc --format json`、`pcbnew` Python API（DSN 匯出、ratsnest、幾何遍歷）
- **Freerouting 2.x**：headless CLI（Java 21），Docker 平行化
- **PyTorch + PyG**：Model A（GNN 分支）；UNet 用純 PyTorch
- **shapely**：reference plane crossing、loop area、製造性幾何檢查
- **ngspice + PySpice**：schematic 階段功能正確性（既定）
- **FEniCSx**：Tier 1.5 IR drop、Phase 4 熱/PI ground truth（既定）

## 9. 增補決策：無預算限制版（2026-06-11）

前提改變：算力/預算不設限。原則不變——**確定性檢查永遠不換成模型，GRPO 也不需要 critic/value model**。預算花在四個方向，優先序如下：

### 9.1 真值比例拉滿（最高 ROI，先做）

預算最該花的地方不是更多神經網路，而是**更多 ground truth**——標籤永遠比模型貴：

- Freerouting 從「top-K 抽查」升級為 **100% 非同步全量驗證**：每個 rollout 都丟進 CPU farm 跑真佈線，結果非同步回灌 surrogate 訓練集。surrogate 的角色從「替代真值」變成「隱藏延遲」——RL 迴圈用 surrogate 即時給 reward 不被 30–60s 卡住，真值在背景持續校準。
- Tier 2 資料產線開大：openEMS（單次幾分鐘~小時）、FEniCSx 批次跑，目標是讓 Model B 家族的訓練集不再是瓶頸。

### 9.2 Model A 變 ensemble（3–5 顆異質模型）

同任務、不同歸納偏置：GNN 為主、CNN/UNet 為主、graph transformer、不同 seed/資料切分。

- Reward 用 **ensemble 平均**；
- **Disagreement（變異數）當 uncertainty 訊號**：高分歧的 rollout 優先送真值驗證（uncertainty-gated audit）。這是 model-based RL / RLHF reward model ensemble 的標準防 hacking 手段——policy 鑽進的分佈外區域，正是各模型意見分歧之處。

### 9.3 Model B 提前、拆成專科家族

不做一顆多工物理模型——各 solver 的資料產出速率與分佈差太多，拆開各自獨立訓練與更新：

| 模型 | 任務 | Ground truth | 形態 | 備註 |
|------|------|--------------|------|------|
| B1 | 串擾 / SI（NEXT/FEXT、阻抗不連續） | openEMS FDTD | 耦合 track 段對的 GNN/transformer | 主力，closed-form 之外最缺的訊號 |
| B2 | PDN IR drop | FEniCSx DC | UNet over 銅箔 raster | **可能不用訓**：先驗證 Tier 1.5 的 2D 快解延遲，多層 via 陣列超標才訓 |
| B3 | 熱（元件溫度場） | FEniCSx heat | UNet/GNN | 直接接 AIO 熱設計領域知識 |
| B4 | EMI 輻射（far-field） | openEMS far-field | — | 最難、標籤最貴，排最後 |

### 9.4（選配、有風險）Realism discriminator

訓一顆判別器分辨「真人板 vs 生成板」，抓「指標全過但人類不會接受」的退化解（元件邏輯分群、絲印可讀性、慣例佈局）。風險是經典 discriminator hacking——**權重必須小、只當 shaping、且與 9.2 的 audit 機制並用**。不確定有沒有必要，等 RL 跑起來觀察退化模式再決定。

### 即使預算無限也不做的事

- 學習版 DRC / unconnected / 長度匹配檢查（確定性完勝）
- LLM judge 當 reward（不可重現、可被 prompt-level hacking、且慢）
- GRPO 的 critic/value model（演算法本身不需要）

### 對 Phase 排程的影響

- Phase 1 資料引擎同時為 A 與 B 家族服務（openEMS/FEniCSx 批次標註與 Freerouting 並行建）
- Phase 2 直接訓 ensemble（多 seed 是免費的，異質架構多 1–2 週）
- Phase 3 上線即帶 100% 非同步驗證 + uncertainty-gated audit
- Phase 4 從「Model B 開始做」變成「B1/B3 上線」（資料已在 Phase 1 開跑）

## 開放問題（下次討論）

1. LLM 輸出格式定版：直接生 `.kicad_pcb` S-expression，還是生 pcbnew Python script？（檢查器兩者通吃，但影響 tokenizer 效率與 action space）
2. 真實板爬蟲的授權白名單（MIT/CERN-OHL/CC-BY 可用，GPL 板要不要進訓練集）
3. Freerouting effort budget 的確切參數（pass 數 vs timeout 的取捨會影響標籤雜訊）
