"""資料引擎（Phase 1）：四條資料流 × 四條標註產線。

設計見 docs/data-engine.md。實作順序：
1. Stream P 程序化生成（電路模板文法 + placement 合成器）
2. L-route Freerouting 批次標註
3. Stream R GitHub 收集（kiutils 過濾、授權白名單、雙層去重）
4. Stream X 擾動增強
"""
