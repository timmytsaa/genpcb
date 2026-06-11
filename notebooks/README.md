# notebooks/

紀律（docs/generator-model-selection.md §3.5）：

- **只消費 artifacts、不生產**：資料來源限 `experiments/` 與 W&B。
- 不在 notebook 裡訓練。SFT/GRPO 一律 `python -m genpcb.train.*` 跑 script。
- 煙霧測試階段可用單一參數化 notebook 當啟動器（papermill 式），跑完即棄。
