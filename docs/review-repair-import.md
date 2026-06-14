# 審查 / 修改 / 匯入：整合路線（design note）

> 狀態：方向已定（2026-06-14），尚未實作。回答「未來要匯入檔案/圖片確認 PCB layout/component
> 是否需修改，要不要加進目前訓練」——**結論：不加進現在的 placement GRPO**，但架構已預留接口。
> 上游：reward/checker（[placement-routing-checker.md](placement-routing-checker.md)）、IR/DSL（[output-format.md](output-format.md)）、資料引擎 Stream X（[data-engine.md](data-engine.md) §1.3）。

---

## 0. 一句話

「這塊板需不需要改」**主要不是一個要訓練的任務，而是 checker 本身**。匯入檔案靠共用 parser、修改靠 repair（資料已備）、圖片是另一條獨立 vision 線。現在的訓練一行都不用改。

---

## 1. 審查 = checker，不是新模型

- 「需不需要修改」的判斷 = `compute_reward(board)` / `placement_metrics(board)`：reward 低、或某指標
  （overlap / 越界 / DRC / decap 過遠 / congestion）fail，**就是「需要修改」且「哪裡要改」**。
- checker 對**任何**板都能跑（生成的或匯入的），只要先轉成 IR。
- 確定性審查 **比 LLM 評論可靠**（可重現、不會幻覺）。→ 不為了 review 另訓一顆模型。

## 2. 匯入：檔案 vs 圖片是兩條不同的線

| 輸入 | 難度 | 做法 | enabler |
|------|------|------|---------|
| 檔案（.kicad_pcb 等） | 低 | parse → IR → 跑 checker | `.kicad_pcb ↔ IR` parser（已在 roadmap，與 reward 共用） |
| 圖片（渲染圖/實物照） | 高 | 視覺逆向：圖 → 結構化 layout | 獨立 vision pipeline（object detection / OCR / 多模態），**不塞進 placement GRPO** |

檔案匯入幾乎免費（共用 parser）；圖片匯入是獨立大工程，之後單獨開。Gemma 4 的 encoder-free
多模態（generator-model-selection §2.5）對圖片線有利，但仍是另一條 track。

## 3. 「自動修改」= board repair = 換條件的 generation

- 任務形態：輸入 = 現有（有問題的）板，輸出 = 改良的板。與現在 placement 同家族（只是 prompt
  多帶「現有擺位」）。
- **訓練資料已經在產**：資料引擎 Stream X 的擾動增強產出 (壞板 → 好板) 配對，正好是 repair 素材。
  以「壞板為輸入、好板為目標」重用即可。
- 因此 repair 是未來一個自然階段，用同一顆模型多任務，不另起爐灶。

## 4. 對話式 critique（選配）

若要「解釋為何要改 C3」這種自然語言評論，是獨立 SFT 任務（board → critique text），可多任務掛
同一顆模型。優先序低於上面三項，且 critique 的「事實依據」仍來自 checker（避免幻覺）。

## 5. 為什麼現在不加

1. GRPO 才從冷啟動崩潰救回來、正在驗證穩定性；此時加第二任務會糊掉好不容易穩住的 placement 訓練。**一次一個任務。**
2. 架構已預留：IR/DSL 當共同交換格式、checker 當審查引擎、Stream X 當 repair 素材。現在訓練不用動，未來接得上。

## 6. 整合順序（排進 roadmap）

```
現在   placement GRPO 跑穩（進行中）
P0     .kicad_pcb ↔ IR parser   ← 同時解鎖 routing reward + 檔案匯入審查（一石二鳥）
P1     board repair 任務（用 Stream X 擾動資料）= 「自動修改」
P2     圖片 → 結構 vision track（獨立，最大工程）
P3     對話式 critique（獨立 SFT，可多任務掛同顆模型）
```

最高槓桿是 **P0 的 parser**：它本來就是 routing reward 的前置，順帶就把「檔案匯入審查」解鎖了。

**P0 進度（2026-06-14）**：`genpcb/kicad/` 已實作純 Python 雙向橋——`board_to_kicad_pcb`
（IR → 自成一檔的 .kicad_pcb，內嵌近似 pad 幾何）、`kicad_pcb_to_board`（解析回 IR）。
45 板 round-trip（components/nets/板框）100% 一致。**待 KiCad 環境**：pcbnew/kicad-cli 載入
驗證、Freerouting 佈線（routing reward）、真實庫 footprint 名→型號對應（file-import 用）。

## 7. 對現在架構的要求（確認已滿足）

- IR/DSL 維持為唯一交換格式（生成、審查、修改都走它）✓
- checker 以 IR 為輸入、與「板從哪來」解耦 ✓
- Stream X 擾動配對保留 (壞→好) 對應關係（repair 用）→ 資料引擎產製時記得保留來源 anchor 連結
