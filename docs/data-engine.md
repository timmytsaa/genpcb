# 訓練資料引擎設計

> 狀態：已決策（2026-06-11）。服務對象：Model A ensemble（routability）、Model B 家族（B1 串擾 / B2 IR drop / B3 熱）、realism discriminator（選配）、以及生成器的 SFT warm-start。
> 前提：無預算限制——預算主力倒在標註產線（ground truth），不是模型。

---

## 0. 總覽：四條板來源資料流 × 四條標註產線

```
板來源 (Streams)                標註產線 (Labelers)            資料集 (per model)
┌─ P 程序化生成 ──┐
├─ R 真實板收集 ──┤──┬→ L-route  Freerouting批次  →  D-A   (Model A ensemble)
├─ X 擾動增強  ───┤  ├→ L-em     openEMS 耦合結構  →  D-B1  (串擾/SI)
└─ D RL rollout ──┘  ├→ L-dc     FEniCSx DC       →  D-B2  (IR drop)
                     └→ L-th     FEniCSx heat     →  D-B3  (熱)
                     R 的人類佈線原版 ────────────→  D-disc (discriminator 正樣本)
                     R+P 的板表示序列 ────────────→  D-sft  (生成器 warm-start)
```

所有 stream 輸出統一中間格式：`(netlist graph, placement, [routing], board meta, 來源標記)`，labeler 只認這個格式，stream 與 labeler 完全解耦——RL 上線後 rollout（Stream D）直接走同一條產線。

---

## 1. 板來源：四條資料流

### 1.1 Stream P：程序化生成（覆蓋度主力）

**電路模板文法**——不是隨機亂接，是參數化電路家族的組合文法，ERC-clean by construction：

| 家族 | 內容 | 練到的 pattern |
|------|------|----------------|
| MCU 板 | MCU + decoupling 群 + 晶振 + SWD + USB + LDO | decap 鄰近性、晶振短走線 |
| Buck converter | controller + 功率級 + 回授 | 大電流環路、熱、敏感回授 |
| 類比前端 | op-amp 濾波/放大鏈 | 類比分區、guard |
| Connector breakout | 高 pin 數連接器扇出 | 板邊約束、bus 長度匹配 |
| 感測器板 | I2C/SPI 多裝置 | 匯流排拓撲 |
| FPGA-lite | QFP/QFN 高扇出 + 記憶體 | 密集 escape routing |

- **組合規則**：電源段 × 1 + 核心家族 × 1 + 周邊家族 × 0–3，透過介面 net（電源軌、bus）拼接。一個模板實例化時取樣：元件值/封裝（KiCad 官方庫，0402–0805 被動、SOIC/QFP/QFN/小 BGA）、net class 指定（diff pair、power）。
- **板參數**：板框面積 vs 元件總 courtyard 面積的比值是**密度旋鈕**（取樣 1.5×–4×）、層數（2/4）、design rule 等級、connector 鎖板邊。
- 產出後跑 ERC + 空板 DRC 守門，不過就丟棄（記錄丟棄率，>5% 表示文法有 bug）。

**Placement 合成器（Model A 資料的心臟）**——一張 netlist 要產出**品質光譜**，每張 10–20 個 placement：

| 合成器 | 方法 | 預期品質 |
|--------|------|----------|
| `random_valid` | 隨機撒 + courtyard 無重疊 rejection | 差 |
| `force_directed` | 連線當彈簧鬆弛 | 中 |
| `sa_hpwl` | Simulated annealing 最小化 HPWL（含鄰近規則項） | 好（合成上限 anchor） |
| `sa_hpwl` 早停 | SA 在不同溫度截斷 | 中–好的連續譜 |

同 netlist 的這組 placement 天然構成 ranking 群組（對齊 GRPO group 結構），這是 §3 切分紀律的單位。

### 1.2 Stream R：真實板收集（分佈真實性主力）

```
GitHub API 搜尋 .kicad_pcb（按 repo）
  → 授權白名單過濾：MIT / BSD / Apache / CERN-OHL-P/W / CC-BY（GPL/CC-BY-SA 隔離存放，僅用於評測不進訓練）
  → kicad-cli 批次升版到 KiCad 9 格式（5/6/7/8 全部正規化）
  → kiutils 純 Python 解析做粗過濾（不用裝 KiCad、可大規模平行）：
      元件數 5–300、層數 2–4、無 flex/異形工藝、netlist 完整
  → 去重：netlist 拓撲 hash + footprint 集合 hash（fork/抄板極多，repo 級 + 板級雙層去重）
  → 拆 tracks/vias → (netlist, 人類 placement) ；原始佈線版本另存
```

真實板的三重用途：
1. 人類 placement = 高品質 anchor（比 `sa_hpwl` 更真）
2. 人類 routing 原版 = `wl_ratio` 校準基準 + discriminator 正樣本 + 生成器 SFT 資料
3. netlist 餵給 placement 合成器再生變體

工具決策：**收集/過濾用 kiutils（純 Python、快、好平行），標註用 pcbnew（Docker 內 kicad:9.x image，因為 DSN 匯出與 ratsnest 只有它有）**。

### 1.3 Stream X：擾動增強（廉價品質光譜）

對 P 的 `sa_hpwl` anchor 與 R 的人類 placement 施加退化算子：

| 算子 | 參數 | 模擬的失敗模式 |
|------|------|----------------|
| 位置 jitter | σ ∈ {1, 5, 20} mm | 普遍亂度 |
| 旋轉隨機化 | 90° 倍數 / 任意 | 方向錯誤 |
| 元件對調 | 同尺寸 / 跨尺寸 | 邏輯分群破壞 |
| Cluster 打散 | 把連通子圖拆散到遠處 | decap 遠離、分區破壞 |
| 鏡像/換層 | SMT 換面 | 層指定錯誤 |
| 板框壓縮 | 縮板框逼高密度 | 壅塞極端化 |

強度做成連續排程（每板取樣 3–5 個強度點），讓品質光譜平滑、ranking 訊號處處有梯度。

### 1.4 Stream D：RL rollout 回灌（上線後的主流量）

RL 開跑後 policy 生成的板走 100% 非同步真值驗證（§9.1 既定決策），驗證結果自動 append 進 D-A。Schema 從第一天就相容，無需改造。**注意：Stream D 樣本要帶 policy 版本標記**，分析 surrogate 漂移時按版本切。

---

## 2. 標註產線

### 2.1 L-route：Freerouting 批次（→ D-A）

```
placement.kicad_pcb
  → pcbnew.ExportSpecctraDSN()
  → freerouting headless（固定 budget：pass 上限 + 60s timeout，參數定版後不再動）
  → .ses 匯回 pcbnew
  → 抽取標籤：routed_fraction / per_net_routed / wl_ratio / via_count /
              congestion_map（佈線後銅密度 + 每網路繞行比 detour ratio 的 2D 圖）/
              匯回後 DRC 違規數
```

- 平行化：Docker 多容器、每容器一 JVM，純 CPU 任務線性擴展。
- **標籤雜訊測量（必做）**：抽 500 板每板重跑 5 次，量 routed_fraction 變異數 → 這個數字決定 ranking loss 的 margin 與「兩個 placement 算不算同分」的閾值。若雜訊過大，驗證集標籤改用 3 次取中位數。
- Budget 參數是標籤定義的一部分——改 budget = 換資料集版本。

### 2.2 L-em：openEMS（→ D-B1）

**關鍵決策：不模擬整板**。整板 FDTD 一次數小時且學不到可遷移結構；改成兩路：

1. **參數掃描（覆蓋度）**：合成耦合 microstrip/stripline 結構——2–4 條線，掃 線寬/間距/平行長度/層疊/端接，FDTD 出 S-params → NEXT/FEXT。單次 2–10 分鐘。
2. **真板裁切（真實性）**：從已佈線板（R 原版 + L-route 產出）裁出含耦合段的局部區域（含參考平面實況、plane split），模擬同樣結構。讓 B1 看得懂真實板的髒環境。

B1 的輸入表示因此是「耦合 track 段組 + 局部層疊/平面上下文」，不是整板——推理時掃過整板所有耦合段組。

### 2.3 L-dc：FEniCSx DC（→ D-B2）

銅箔 polygon → gmsh 網格 → 2D sheet-resistance Laplace、via 做層間導通、元件功率腳當電流源/汲 → IR drop map + 電流密度熱點。單次次秒級～數秒，**先用它驗證 Tier 1.5 的 in-loop 延遲，達標就不用訓 B2**（既定保留決策）。

### 2.4 L-th：FEniCSx heat（→ D-B3）

- 功率模型：每元件類型掛功率分佈（LDO/MOSFET/MCU 各自的典型耗散範圍）取樣，這本身是資料增強維度。
- 2.5D 板級熱傳（銅當 spreader、等效對流邊界）→ 溫度場 + 每元件 Tmax。
- 32 吋 AIO 那套雙路徑熱規格經驗直接用在邊界條件與驗證 case 設計上。

### 2.5 成本估算（256 核 farm 基準）

| 產線 | 單次 | 量 | 總 CPU 時間 | farm 牆鐘 |
|------|------|----|------------|-----------|
| L-route | ~30 s | 375k placement | ~130 CPU-day | ~12 hr |
| L-em | ~5 min | 50k 結構 | ~175 CPU-day | ~1 day（記憶體上限酌減併發） |
| L-dc | ~2 s | 200k | ~5 CPU-day | 小時級 |
| L-th | ~30 s | 100k | ~35 CPU-day | ~3 hr |

全部加起來一週內可完成第一輪。瓶頸不在算力在工程：Freerouting 容錯（卡死板要 timeout kill）、openEMS 網格失敗重試、損壞 .kicad_pcb 的防禦性解析。

---

## 3. 資料集規格與切分紀律

- **儲存**：訓練樣本（graph tensors + raster + labels）用 WebDataset/Parquet 分片；**原始 .kicad_pcb 永久保留**（表示法改版要能重抽取）。
- **樣本 schema 帶 provenance**：stream 來源、模板 id、擾動算子+強度、labeler 版本、（Stream D）policy 版本。所有後續分析靠這個。
- **切分紀律（防洩漏，最容易犯的錯）**：
  - 以 **netlist/設計** 為切分單位——同 netlist 的所有 placement 變體必須同 split；
  - 真實板以 **repo** 為單位切（fork/抄板會跨 repo 近重複，靠 1.2 的雙層去重兜底）；
  - 程序化板以 **模板家族 × 取樣 seed 區段** 切，並保留 1–2 個家族完全不進訓練當 OOD 測試。
- **版本化**：資料集不可變快照 + append-only 的 Stream D 增量；surrogate 重訓記錄對應快照 hash。

---

## 4. 資料引擎自我驗證（資料的驗收門檻）

1. **標籤雜訊報告**（2.1 的重跑實驗）→ 定 ranking margin。
2. **分佈對齊報告**：程序化板 vs 真實板的元件數/密度/net degree/扇出分佈 QQ 圖——程序化文法漂進玩具區就要修模板，否則 Model A 在真實板上失效。
3. **品質光譜覆蓋**：每個 ranking 群組內 routed_fraction 的分佈要鋪開（全是 1.0 或全是 0.3 的群組沒有 ranking 訊號，回頭調擾動強度排程）。
4. **守門統計**：ERC/DRC-clean 率、解析失敗率、labeler 失敗率，全部進 dashboard。

---

## 5. 與生成器 SFT 的共用

同一個 corpus 餵兩邊：真實板（R）+ 高品質合成板（P 的 `sa_hpwl` + L-route 佈通版）序列化成生成器的目標格式，就是 Qwen2.5-Coder 的 SFT warm-start 資料。**這把「LLM 輸出格式定版」的開放問題變得更急**——格式定了，資料引擎一次跑就同時產 checker 與 generator 兩邊的資料。

---

## 6. 量化目標（第一輪）

| 項目 | 目標 |
|------|------|
| 程序化 netlist | 20k（六家族 × 參數取樣） |
| 真實板（去重後） | 5–10k |
| Placement 樣本（D-A） | 300–500k（每 netlist 10–20 變體） |
| D-B1 耦合結構 | 50k |
| D-B3 熱樣本 | 100k |
| OOD 保留 | 2 個模板家族 + 10% 真實 repo |
