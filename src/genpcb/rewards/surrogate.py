"""Surrogate-driven placement reward（GRPO in-loop）。

決策（討論定案）：routability surrogate 當主品質訊號、overlap/越界當硬 gate、HPWL 拿掉。
surrogate 推理純 CPU、~ms，可每 rollout 即時算。真值由 routing_reward 背景 audit（DAgger）。
"""

from __future__ import annotations

import torch

from genpcb.data.procedural import Board
from genpcb.rewards.metrics import placement_metrics
from genpcb.surrogate.features import board_to_graph, board_to_raster
from genpcb.surrogate.model import RoutabilitySurrogate

# w_route 主導品質；overlap/bounds 為硬 gate；decap 輕領域；無 HPWL
DEFAULT_WEIGHTS = {"route": 5.0, "overlap": 8.0, "bounds": 8.0, "decap": 0.5}


def load_surrogate(path: str, hidden: int = 64) -> RoutabilitySurrogate:
    model = RoutabilitySurrogate(hidden=hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def surrogate_routability(board: Board, model: RoutabilitySurrogate) -> float:
    """surrogate 預測 routed_fraction ∈[0,1]。"""
    g = board_to_graph(board)
    r = board_to_raster(board)
    with torch.no_grad():
        return float(model(
            torch.tensor(g["x"], dtype=torch.float32),
            torch.tensor(g["edge_index"], dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
        ))


def surrogate_reward(board: Board, model: RoutabilitySurrogate, weights: dict | None = None) -> tuple[float, dict]:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    m = placement_metrics(board)
    n = max(1, len(board.components))
    rf = surrogate_routability(board, model)
    reward = (
        w["route"] * rf                                 # 主品質訊號（routability）
        - w["overlap"] * m["overlap_area"]              # 硬 gate
        - w["bounds"] * (m["oob_count"] / n)            # 硬 gate
        - w["decap"] * m["decap_penalty"]               # 輕領域 shaping
    )
    return reward, {"surrogate_rf": rf, **m}
