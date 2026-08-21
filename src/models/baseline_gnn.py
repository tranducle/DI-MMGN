"""
Baseline GNN Models (GraphSAGE / GAT / GIN)
Paper ref: paper_outline.md §5.2.
Static snapshot classifier that takes a single DOM graph (t2) and predicts 
defaced (1) vs legitimate (0). It serves as the non-temporal, non-orthogonal baseline.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from .encoders import DOMGraphEncoder


class _GATStack(nn.Module):
    """3-layer GAT + global mean pool + linear -> d. Mirror of DOMGraphEncoder."""
    def __init__(self, d=256, in_dim=70, hidden=128, layers=3, heads=2):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        cur = in_dim
        for _ in range(layers):
            # heads*out must match next in; keep hidden = heads*out_per_head
            out_per_head = hidden // heads
            self.convs.append(GATConv(cur, out_per_head, heads=heads))
            cur = out_per_head * heads
        self.pool = global_mean_pool
        self.proj = nn.Linear(cur, d)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        h = self.pool(h, batch)
        return self.proj(h)


class _GINStack(nn.Module):
    """3-layer GIN (MLP aggregator) + global mean pool + linear -> d."""
    def __init__(self, d=256, in_dim=70, hidden=128, layers=3):
        super().__init__()
        from torch_geometric.nn import GINConv
        self.convs = nn.ModuleList()
        cur = in_dim
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(cur, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
            cur = hidden
        self.pool = global_mean_pool
        self.proj = nn.Linear(cur, d)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        h = self.pool(h, batch)
        return self.proj(h)


class _StaticBaseline(nn.Module):
    """Shared static-snapshot classifier shell: encoder + MLP head."""
    def __init__(self, encoder, d=256):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))

    def forward(self, snap_t2):
        feats = self.encoder(snap_t2["dom_x"], snap_t2["dom_edge_index"], snap_t2["dom_batch"])
        return self.classifier(feats).squeeze(-1)


class BaselineGraphSAGE(nn.Module):
    """Static snapshot classifier baseline using GraphSAGE."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg["model"].get("d", 256)
        hidden = cfg["model"].get("dom_hidden", 128)
        layers = cfg["model"].get("dom_layers", 3)
        self.encoder = DOMGraphEncoder(d=d, in_dim=70, hidden=hidden, layers=layers)
        self.classifier = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))

    def forward(self, snap_t2):
        feats = self.encoder(snap_t2["dom_x"], snap_t2["dom_edge_index"], snap_t2["dom_batch"])
        return self.classifier(feats).squeeze(-1)


class BaselineGAT(nn.Module):
    """Static snapshot classifier baseline using GAT (graph attention)."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg["model"].get("d", 256)
        hidden = cfg["model"].get("dom_hidden", 128)
        layers = cfg["model"].get("dom_layers", 3)
        shell = _StaticBaseline(_GATStack(d=d, in_dim=70, hidden=hidden, layers=layers), d=d)
        self.encoder = shell.encoder
        self.classifier = shell.classifier

    def forward(self, snap_t2):
        feats = self.encoder(snap_t2["dom_x"], snap_t2["dom_edge_index"], snap_t2["dom_batch"])
        return self.classifier(feats).squeeze(-1)


class BaselineGIN(nn.Module):
    """Static snapshot classifier baseline using GIN (graph isomorphism)."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg["model"].get("d", 256)
        hidden = cfg["model"].get("dom_hidden", 128)
        layers = cfg["model"].get("dom_layers", 3)
        shell = _StaticBaseline(_GINStack(d=d, in_dim=70, hidden=hidden, layers=layers), d=d)
        self.encoder = shell.encoder
        self.classifier = shell.classifier

    def forward(self, snap_t2):
        feats = self.encoder(snap_t2["dom_x"], snap_t2["dom_edge_index"], snap_t2["dom_batch"])
        return self.classifier(feats).squeeze(-1)


BASELINE_REGISTRY = {
    "graphsage": BaselineGraphSAGE,
    "gat": BaselineGAT,
    "gin": BaselineGIN,
}
