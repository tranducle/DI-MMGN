"""Per-modality encoders + shared fusion E(.) for DI-MMGN.

Paper ref: paper_outline.md §4.1 — snapshot S_t = (text, DOM-graph, visual, HTTP),
shared encoder E(.) -> R^d, change vector Δ_t = E(S_t) - E(S_{t-1}).

Modality-mask-weighted fusion respects the dataset's asymmetric coverage (only
~1,971 CMS eval pairs are 4-modality; the rest are text+HTTP). Missing modalities
are masked out, never silently zeroed.
"""
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Frozen miniLM (384-d) + linear projection -> d."""
    def __init__(self, d=256, backbone="sentence-transformers/all-MiniLM-L6-v2", frozen=True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(backbone)
        self.bert = AutoModel.from_pretrained(backbone)
        self.proj = nn.Linear(384, d)
        if frozen:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]          # [B,384]
        return self.proj(cls)                       # [B,d]


class DOMGraphEncoder(nn.Module):
    """3-layer GraphSAGE + global mean pool + linear -> d. Paper §5.2 names SAGE."""
    def __init__(self, d=256, in_dim=70, hidden=128, layers=3):
        super().__init__()
        from torch_geometric.nn import SAGEConv, global_mean_pool
        self.convs = nn.ModuleList()
        cur = in_dim
        for _ in range(layers):
            self.convs.append(SAGEConv(cur, hidden))
            cur = hidden
        self.pool = global_mean_pool
        self.proj = nn.Linear(hidden, d)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        h = self.pool(h, batch)                     # [B,hidden]
        return self.proj(h)                          # [B,d]


class VisualEncoder(nn.Module):
    """CLIP-512 embedding -> 2-layer MLP -> d."""
    def __init__(self, d=256, in_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        return self.mlp(x)                           # [B,d]


class HTTPEncoder(nn.Module):
    """32-d HTTP-signal vector -> 2-layer MLP -> d."""
    def __init__(self, d=256, in_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        return self.mlp(x)                           # [B,d]


class MultiModalEncoder(nn.Module):
    """Mask-weighted fusion of the four modality encoders -> shared E(.) -> R^d.

    mask: [B,4] bool, order = (text, dom, visual, http). Missing modalities are
    excluded from the mean so the fusion never silently treats them as zero evidence.

    Dropout (cfg["model"]["dropout"], default 0.0) is applied at the fusion output.
    p=0.0 reproduces the original Phase-4 model bit-for-bit; p>0 introduces the
    persistent training stochasticity needed for honest seed-variance CIs
    (see 8_Project_Management/seeding_audit.md).
    """
    def __init__(self, cfg):
        super().__init__()
        d = cfg["model"]["d"]
        self.text = TextEncoder(d, frozen=cfg["model"]["text_frozen"])
        self.dom = DOMGraphEncoder(d, in_dim=70, hidden=cfg["model"]["dom_hidden"],
                                   layers=cfg["model"]["dom_layers"])
        self.visual = VisualEncoder(d, in_dim=cfg["model"]["visual_in"])
        self.http = HTTPEncoder(d, in_dim=cfg["model"]["http_in"])
        self.drop = nn.Dropout(float(cfg["model"].get("dropout", 0.0)))

    def forward(self, snap, mask):
        # snap: dict with text{input_ids,attention_mask}, dom{x,edge_index,batch}, visual, http
        e_text = self.text(snap["text_input_ids"], snap["text_attention_mask"])
        e_dom = self.dom(snap["dom_x"], snap["dom_edge_index"], snap["dom_batch"])
        e_vis = self.visual(snap["visual"])
        e_http = self.http(snap["http"])
        feats = torch.stack([e_text, e_dom, e_vis, e_http], dim=1)   # [B,4,d]
        m = mask.unsqueeze(-1).float()                                # [B,4,1]
        num = (feats * m).sum(1)                                      # [B,d]
        den = m.sum(1).clamp(min=1.0)                                 # [B,1]
        return self.drop(num / den)                                   # [B,d]
