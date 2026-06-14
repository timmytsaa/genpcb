"""分層 reward（docs/placement-routing-checker.md）。

兩半、環境不同：
- **Tier 1 placement（純 Python，每 rollout 即時）**：metrics.py 的 courtyard
  重疊 / 越界 / HPWL / RUDY / decap 距離。本檔 `placement_reward` 聚合，
  形態 A 的 GRPO 主 reward。
- **Tier 1 routing + DRC / Freerouting / surrogate（需 KiCad 環境）**：
  `compute_reward(board_path=...)` 之路，尚未實作（dsl→.kicad_pcb 還原器是前置）。
"""

from __future__ import annotations

from genpcb.data.procedural import Board
from genpcb.data.serialize import dsl_to_board, sft_example_to_dsl
from genpcb.rewards.metrics import placement_metrics

_ROT = {"0", "90", "180", "270"}
_SIDE = {"T", "B"}

# 預設權重；正式跑由 config 的 reward 區覆蓋（見 configs/base.yaml）
DEFAULT_WEIGHTS = {
    "w_overlap": 8.0,
    "w_bounds": 8.0,
    "w_hpwl": 1.0,
    "w_rudy": 0.0,    # 未校準，暫不入 reward（見 placement_reward 註解）
    "w_decap": 0.5,
    "rudy_bins": 32,
}


def placement_reward(board: Board, weights: dict | None = None) -> tuple[float, dict]:
    """形態 A 的 Tier-1 reward。回傳 (reward, breakdown)。

    硬違規（overlap/bounds）權重大，主導；連續項（hpwl/rudy/decap）做 shaping。
    reward ≤ 0，0 = 無違規且線長/壅塞為零的理想下界。
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    m = placement_metrics(board, rudy_bins=int(w["rudy_bins"]))
    n = max(1, len(board.components))
    # 全部用有界項，使硬違規（overlap/越界比例）主導、防「全疊一點」式 reward hacking。
    # rudy_peak 未校準且無界 → 預設權重 0（仍算出供 audit），HPWL 暫代壅塞 proxy。
    reward = -(
        w["w_overlap"] * m["overlap_area"]            # ≈[0,2]：重疊面積 / 元件總面積
        + w["w_bounds"] * (m["oob_count"] / n)         # [0,1]：越界元件比例（有界）
        + w["w_hpwl"] * m["hpwl_norm"]                 # ≈[0,1.5]
        + w["w_rudy"] * m["rudy_peak"]                 # 預設 0；校準後再開
        + w["w_decap"] * m["decap_penalty"]            # 領域規則
    )
    return reward, m


def _board_from_completion(prompt: str, completion: str) -> Board:
    """嚴格解析 policy 輸出：completion 必須是良構且**完整**的擺位。

    任一結構問題（壞行 / 未宣告 ref / 重複 / 壞 rot/side / 非整數座標 /
    未擺滿所有宣告元件）都 raise ValueError → 由呼叫端給 floor 懲罰。
    完整性檢查很重要：否則「少擺幾個元件」會壓低 HPWL 形成 reward hacking。
    """
    declared: dict[str, str] = {}
    for line in prompt.splitlines():
        t = line.split()
        if t and t[0] == "D":
            declared[t[1]] = t[2]
    placed: set[str] = set()
    for line in completion.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] != "P" or len(t) != 6:
            raise ValueError(f"bad placement line: {line!r}")
        _, ref, x, y, rot, side = t
        if ref not in declared:
            raise ValueError(f"undeclared ref: {ref}")
        if ref in placed:
            raise ValueError(f"duplicate ref: {ref}")
        int(x); int(y)                       # 非整數 → ValueError
        if rot not in _ROT or side not in _SIDE:
            raise ValueError(f"bad rot/side: {rot}/{side}")
        placed.add(ref)
    if placed != set(declared):
        raise ValueError(f"incomplete: placed {len(placed)}/{len(declared)}")
    return dsl_to_board(sft_example_to_dsl(prompt, completion))


def reward_from_completion(prompt: str, completion: str, weights: dict | None = None,
                           parse_fail: float = -100.0) -> float:
    """GRPO 的 (prompt, completion) → reward 橋。malformed/incomplete → floor。

    floor 比任何合法板都低（合法板 reward 多在 [-30, 0]），讓 policy 先學會輸出
    良構完整的 DSL，再於合法解之間最佳化品質。
    """
    try:
        board = _board_from_completion(prompt, completion)
    except ValueError:
        return parse_fail
    return placement_reward(board, weights)[0]   # 真實 reward 的例外不吞，讓 bug 浮現


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def shaped_reward_from_completion(prompt: str, completion: str, weights: dict | None = None,
                                  invalid_base: float = -50.0, invalid_span: float = 20.0) -> float:
    """GRPO 用的「有梯度」reward：對 malformed/incomplete 給部分分數。

    二元 floor（reward_from_completion）在「整組都壞」時 advantage=0、無梯度，是 GRPO
    冷啟動崩潰的主因。此版改為：
    - 完全合法且完整 → placement_reward（品質分，∈ 約 [-27, 0]）
    - 否則 → invalid_base + invalid_span·完整度 − 壞行數（≤ -30，恆低於任何合法板）

    → 即使整組都不合法，「擺對較多元件」者分數較高，policy 永遠有往合法爬的梯度。
    合法恆勝過不合法（分數帶分隔），故不會獎勵半成品。
    """
    declared: dict[str, str] = {}
    for line in prompt.splitlines():
        t = line.split()
        if t and t[0] == "D":
            declared[t[1]] = t[2]
    n = max(1, len(declared))
    good: set[str] = set()
    malformed = 0
    for line in completion.splitlines():
        t = line.split()
        if not t:
            continue
        if (t[0] == "P" and len(t) == 6 and t[1] in declared and t[1] not in good
                and t[4] in _ROT and t[5] in _SIDE and _is_int(t[2]) and _is_int(t[3])):
            good.add(t[1])
        else:
            malformed += 1
    completeness = len(good) / n
    if completeness == 1.0 and malformed == 0:
        board = dsl_to_board(sft_example_to_dsl(prompt, completion))
        return placement_reward(board, weights)[0]
    return invalid_base + invalid_span * completeness - float(malformed)


def compute_reward(dsl_text: str | None = None, board_path: str | None = None,
                   weights: dict | None = None) -> float:
    """GRPO 迴圈的 reward 入口。

    - dsl_text：形態 A placement DSL → Tier-1 placement reward（已實作）。
    - board_path：.kicad_pcb → routing/DRC/Freerouting reward（需 KiCad，未實作）。
    """
    if dsl_text is not None:
        board = dsl_to_board(dsl_text)
        return placement_reward(board, weights)[0]
    if board_path is not None:
        raise NotImplementedError(
            "routing/DRC reward 需 KiCad 環境（dsl→.kicad_pcb 還原器 + kicad-cli/Freerouting）；"
            "見 docs/placement-routing-checker.md §1。本機無 KiCad，於 reward farm 實作。"
        )
    raise ValueError("compute_reward 需要 dsl_text 或 board_path 其一")
