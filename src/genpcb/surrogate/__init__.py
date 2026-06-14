"""Model A routability surrogate（docs/placement-routing-checker.md §3）。

features.py：Board IR → graph（GNN 用）+ raster（UNet 用），純 numpy、本機可測。
訓練（PyG/torch）在訓練端，分開。
"""

from genpcb.surrogate.features import board_to_graph, board_to_raster

__all__ = ["board_to_graph", "board_to_raster"]
