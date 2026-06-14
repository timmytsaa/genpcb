"""IR ↔ .kicad_pcb 橋（P0）。

- board_to_kicad_pcb：IR → KiCad PCB S-expression（→ Freerouting/DRC routing reward）
- kicad_pcb_to_board：.kicad_pcb → IR（→ 檔案匯入審查、round-trip 驗證）

純 Python，本機可測 round-trip。**最後一哩**（pcbnew/kicad-cli 載入、Freerouting
佈線）需 KiCad 環境驗證，尚未做（見 docs/review-repair-import.md P0）。
"""

from genpcb.kicad.read import kicad_pcb_to_board
from genpcb.kicad.write import board_to_kicad_pcb

__all__ = ["board_to_kicad_pcb", "kicad_pcb_to_board"]
