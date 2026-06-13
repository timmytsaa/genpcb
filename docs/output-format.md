# 生成器輸出格式 v0：Compact Placement DSL

> 狀態：v0 定版（2026-06-11）。這是「開放問題 #1（LLM 輸出格式）」的第一階段答案——
> 涵蓋形態 A（LLM 只做 placement）。形態 B（含 routing）的格式延後到形態 A 收斂後再定。
> 實作：`src/genpcb/data/serialize.py`。

## 為什麼不直接用 .kicad_pcb S-expression

Tokenizer 煙霧測試（docs/generator-model-selection.md §3.7）的硬資料：

- 現代 tokenizer（Qwen3.5 / Gemma 4）對數字做 single-digit splitting，一個座標
  `105.473` ≈ 6.6 tokens。
- 含 pad 幾何 + routing tracks 的原始 .kicad_pcb，40 元件小板就 ~39k tokens，**破 32k context**。

→ 原始 S-expression 對形態 A 直接出局。

## 設計兩原則

1. **只表達 placement + netlist**：footprint 型號隱含 pad 幾何，不展開；不含 routing
   tracks（那是 Freerouting / 形態 B 的事）。
2. **座標格點量化成整數**：預設 0.1mm 格，座標變小整數，消滅小數點與多餘位數。

## 格式

每行一個 record，空白分隔：

```
B <寬> <高> <層數> <grid_mm>          # 板框（寬高為 grid 單位）、層數、格點大小
C <ref> <footprint> <x> <y> <rot> <side>   # 元件：座標為 grid 單位整數，rot∈{0,90,180,270}，side∈{T,B}
N <netname> <ref.pad> <ref.pad> ...   # 連線：net 名 + pin 清單
```

範例（buck，0.1mm 格 → `96 60` 即 9.6mm, 6.0mm）：

```
B 200 140 2 0.1
C U1 SOIC8 96 60 0 T
C L1 IND1210 151 74 0 T
C C1 C0402 184 73 0 T
...
N GND C1.2 U1.1 L1.1 R1.1 J1.1 J2.1
N VOUT C1.1 U1.2 J1.2 J2.2
N N000 L1.2 J2.6 R2.2
```

- 自我描述（header 帶 grid）、可逆：`dsl_to_board(board_to_dsl(b))` round-trip 還原
  （`build.py` 每筆都跑這個自檢）。
- 實測 token 長度（Gemma 4 tokenizer，v0 種子板）：mean ~509、p95 ~1017、max ~1029，
  **全部遠低於 32k**。相對原始 .kicad_pcb 約 75× 壓縮。

## 與檢查器（reward）的橋接

- 生成器輸出 DSL；reward 端要 `.kicad_pcb`（pcbnew/Freerouting/DRC 吃這個）。
- 需要一個 `dsl → .kicad_pcb` 還原器（footprint 由 KiCad 庫實例化、pad 幾何展開）。
  **這是 Phase 0 `pcb_metrics` 的前置工件**，尚未實作（GRPO reward 才需要，SFT 資料產製不需要）。
- 反向 `.kicad_pcb → DSL` 用於把真實板（Stream R）轉成 SFT 目標，待 KiCad 收集管線時實作。

## 待調參數 / 已知取捨

- **grid 大小**：0.1mm 足夠 placement 精度且 token 便宜；若要更省可調 0.2mm，但會犧牲
  精度。等真實板分佈進來再定版。
- **footprint 命名**：v0 用簡短代號（QFN32、C0402…）。對映到 KiCad 真實 footprint
  library 名稱的表，在 dsl→kicad_pcb 還原器一起定。
- **net 名**：訊號 net 用 `Nnnn` 佔位；真實板轉換時保留原始 net 名（對 SI/領域規則有用）。
