"""Model A：routability surrogate（GNN + CNN 融合，純 PyTorch、無 PyG 依賴）。

輸入 = graph（node x [N,9] + edge_index [2,E]）+ raster [4,H,W]；輸出 = routed_fraction∈[0,1]。
GNN 抓連通性、CNN 抓空間壅塞，融合後回歸。小模型、CPU 可訓（板小、樣本少）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGELayer(nn.Module):
    """GraphSAGE-style：self + 鄰居 mean（edge_index 視為無向，雙向聚合）。"""

    def __init__(self, din: int, dout: int):
        super().__init__()
        self.lin_self = nn.Linear(din, dout)
        self.lin_neigh = nn.Linear(din, dout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        agg = torch.zeros(n, x.size(1), device=x.device, dtype=x.dtype)
        deg = torch.zeros(n, 1, device=x.device, dtype=x.dtype)
        if edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
            for s, d in ((src, dst), (dst, src)):           # 雙向
                agg.index_add_(0, d, x[s])
                deg.index_add_(0, d, torch.ones(d.size(0), 1, device=x.device, dtype=x.dtype))
        agg = agg / deg.clamp(min=1.0)
        return F.relu(self.lin_self(x) + self.lin_neigh(agg))


class RoutabilitySurrogate(nn.Module):
    def __init__(self, node_dim: int = 9, raster_ch: int = 4, hidden: int = 64):
        super().__init__()
        self.g1 = SAGELayer(node_dim, hidden)
        self.g2 = SAGELayer(hidden, hidden)
        self.cnn = nn.Sequential(
            nn.Conv2d(raster_ch, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, hidden, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),          # [B, hidden]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, raster: torch.Tensor) -> torch.Tensor:
        """單板（batch=1）：x [N,9], edge_index [2,E], raster [C,H,W] → 純量∈[0,1]。"""
        h = self.g2(self.g1(x, edge_index), edge_index)
        g = h.mean(dim=0)                                   # global mean pool [hidden]
        r = self.cnn(raster.unsqueeze(0)).squeeze(0)        # [hidden]
        return torch.sigmoid(self.head(torch.cat([g, r]))).squeeze()
