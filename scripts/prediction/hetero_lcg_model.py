"""Heterogeneous GNN regressors for literal-clause graphs.

The expected feature dimensions are 7 for positive literals, 8 for negative
literals, 10 for clauses, and 15 for graph-level structural features.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn import (
    HeteroConv,
    SAGEConv,
    GINConv,
    GATConv,
    global_mean_pool,
)
from torch_geometric.utils import softmax


def _safe_nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9):
    """Replace non-finite tensor values with bounded numeric values."""
    if torch.is_tensor(x):
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    return x


def _prepare_struct_x(data, batch_size, struct_in_dim, device):
    """Normalize structural features to shape [batch_size, struct_in_dim]."""
    struct = getattr(data, "struct_x", None)
    if struct is None:
        struct = torch.zeros(batch_size, struct_in_dim, device=device)
    else:
        if struct.dim() == 1:
            struct = struct.view(1, -1)
        struct = struct.to(device)

        # PyG concatenates per-graph structural features into a batch matrix.
        if struct.size(0) == 1 and batch_size > 1:
            struct = struct.repeat(batch_size, 1)
        elif struct.size(0) != batch_size:
            warnings.warn(
                f"struct_x size {struct.size(0)} != batch_size {batch_size}, "
                f"Reshaping and retaining the last {batch_size} rows; verify that the data is aligned"
            )
            struct = struct.view(-1, struct_in_dim)[-batch_size:]

    struct = _safe_nan_to_num(struct, nan=0.0, posinf=1e6, neginf=-1e6)
    return struct


def _get_or_make_batch(data, ntype: str, device: torch.device) -> torch.Tensor:
    """Return `data[ntype].batch` if it exists; otherwise create a single-graph batch vector."""
    store = data[ntype]

    b = getattr(store, "batch", None)
    if torch.is_tensor(b):
        return b.to(device)

    if hasattr(store, "num_nodes") and store.num_nodes is not None:
        num_nodes = int(store.num_nodes)
    elif hasattr(store, "x") and torch.is_tensor(store.x):
        num_nodes = int(store.x.size(0))
    else:
        num_nodes = 0

    return torch.zeros(num_nodes, dtype=torch.long, device=device)


# Pooling modules

class _AttnPool(nn.Module):
    """Graph-level attention pooling for a single node type."""
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if h.numel() == 0:
            return h.new_zeros((1, h.size(-1)))

        score = self.gate(h).squeeze(-1)          # [N]
        alpha = softmax(score, batch)            # [N]
        alpha = alpha.unsqueeze(-1)              # [N, 1]
        out = torch.zeros((int(batch.max().item()) + 1, h.size(-1)), device=h.device, dtype=h.dtype)
        out.index_add_(0, batch, alpha * h)
        return out


class HeteroBaseRegressor(nn.Module):
    """Base heterogeneous GNN regressor shared by the SAGE, GIN, and GAT variants."""

    def __init__(
        self,
        conv_type: str = "hetero_sage",
        pos_in_dim: int = 7,
        neg_in_dim: int = 8,
        clause_in_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        attn_pool_dropout: float = 0.1,
        attn_lambda_init: float = 0.1,
        use_struct: bool = True,
        struct_in_dim: int = 15,
        gat_heads: int = 4,
        strict_input_dim_check: bool = True,
    ):
        super().__init__()
        assert conv_type in {"hetero_sage", "hetero_gin", "hetero_gat"}
        self.conv_type = conv_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attn_pool_dropout = float(attn_pool_dropout)

        # GAT heads are only applicable to the GAT variant.
        if self.conv_type == "hetero_gat":
            self.gat_heads = int(gat_heads)
        else:
            self.gat_heads = None

        self.use_struct = use_struct
        self.struct_in_dim = struct_in_dim
        self.strict_input_dim_check = strict_input_dim_check


        self.pos_in_dim = pos_in_dim
        self.neg_in_dim = neg_in_dim
        self.clause_in_dim = clause_in_dim

        # Learnable attention lambdas
        def _init_logit(p: float) -> float:
            p = float(p)
            if p <= 0.0:
                return -10.0
            if p >= 1.0:
                return 10.0
            return math.log(p / (1.0 - p))

        init_logit = _init_logit(attn_lambda_init)
        self.lambda_pos_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.lambda_neg_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.lambda_clause_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        self.pos_linear = nn.Linear(pos_in_dim, hidden_dim)
        self.neg_linear = nn.Linear(neg_in_dim, hidden_dim)
        self.clause_linear = nn.Linear(clause_in_dim, hidden_dim)


        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(self._build_hetero_conv_layer(hidden_dim))

        self.pos_attn_pool = _AttnPool(hidden_dim, dropout=self.attn_pool_dropout)
        self.neg_attn_pool = _AttnPool(hidden_dim, dropout=self.attn_pool_dropout)
        self.clause_attn_pool = _AttnPool(hidden_dim, dropout=self.attn_pool_dropout)

        pooled_dim = hidden_dim * 6  # mean_attn => 6H

        # Normalize structural embeddings before prediction.
        if self.use_struct:
            self.struct_mlp = nn.Sequential(
                nn.Linear(struct_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            predictor_in_dim = pooled_dim + hidden_dim
        else:
            self.struct_mlp = None
            predictor_in_dim = pooled_dim

        # Normalize hidden layers in the regression head.
        self.predictor = nn.Sequential(
            nn.Linear(predictor_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _build_hetero_conv_layer(self, hidden_dim: int) -> HeteroConv:
        convs = {}

        if self.conv_type == "hetero_sage":
            def make_conv():
                return SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean")

        elif self.conv_type == "hetero_gin":
            def make_conv():
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                return GINConv(mlp)

        else:  # hetero_gat
            def make_conv():
                # Self-loops are disabled for heterogeneous bipartite relations.
                # Residual connections preserve each node type representation.
                return GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=int(self.gat_heads or 1),
                    concat=False,
                    dropout=self.dropout,
                    add_self_loops=False,
                )

        edge_types = [
            ("pos", "to", "clause"),
            ("neg", "to", "clause"),
            ("clause", "to", "pos"),
            ("clause", "to", "neg"),
            ("pos", "to", "neg"),
            ("neg", "to", "pos"),
        ]
        for et in edge_types:
            convs[et] = make_conv()

        return HeteroConv(convs, aggr="sum")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _check_input_dims(self, data):
        if not self.strict_input_dim_check:
            return

        def _dim(x):
            return int(x.size(1)) if torch.is_tensor(x) and x.dim() == 2 else -1

        pos_d = _dim(data["pos"].x)
        neg_d = _dim(data["neg"].x)
        cls_d = _dim(data["clause"].x)

        if pos_d != self.pos_in_dim:
            raise ValueError(f"pos.x dim mismatch: expect {self.pos_in_dim}, got {pos_d}")
        if neg_d != self.neg_in_dim:
            raise ValueError(f"neg.x dim mismatch: expect {self.neg_in_dim}, got {neg_d}")
        if cls_d != self.clause_in_dim:
            raise ValueError(f"clause.x dim mismatch: expect {self.clause_in_dim}, got {cls_d}")

    def forward(self, data):
        self._check_input_dims(data)
        device = data["pos"].x.device

        x_dict = {
            "pos": F.relu(self.pos_linear(data["pos"].x.float())),
            "neg": F.relu(self.neg_linear(data["neg"].x.float())),
            "clause": F.relu(self.clause_linear(data["clause"].x.float())),
        }
        x_dict = {k: _safe_nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6) for k, v in x_dict.items()}

        # Residual connections stabilize message passing across layers.
        for conv in self.convs:
            x_prev = x_dict
            x_dict = conv(x_dict, data.edge_index_dict)
            for ntype in x_dict:
                x = _safe_nan_to_num(x_dict[ntype], nan=0.0, posinf=1e6, neginf=-1e6)
                if ntype in x_prev and x_prev[ntype].shape == x.shape:
                    x = x + x_prev[ntype]  # residual
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x_dict[ntype] = x

        batch_pos = _get_or_make_batch(data, "pos", device)
        batch_neg = _get_or_make_batch(data, "neg", device)
        batch_clause = _get_or_make_batch(data, "clause", device)

        # Prefer the batch size supplied by the PyG DataLoader.
        bs = getattr(data, "batch_size", None)
        if isinstance(bs, int) and bs > 0:
            batch_size = bs
        else:
            if batch_pos.numel() > 0:
                batch_size = int(batch_pos.max().item()) + 1
            elif batch_neg.numel() > 0:
                batch_size = int(batch_neg.max().item()) + 1
            elif batch_clause.numel() > 0:
                batch_size = int(batch_clause.max().item()) + 1
            else:
                batch_size = 1

        def _mean_attn_pool(h: torch.Tensor, batch: torch.Tensor, ntype: str) -> torch.Tensor:
            mean_p = global_mean_pool(h, batch)
            if ntype == "pos":
                attn_p = self.pos_attn_pool(h, batch)
                lam = torch.sigmoid(self.lambda_pos_logit)
            elif ntype == "neg":
                attn_p = self.neg_attn_pool(h, batch)
                lam = torch.sigmoid(self.lambda_neg_logit)
            else:
                attn_p = self.clause_attn_pool(h, batch)
                lam = torch.sigmoid(self.lambda_clause_logit)

            attn_p = attn_p * lam

            if mean_p.size(0) < batch_size:
                mean_p = torch.cat([mean_p, mean_p.new_zeros((batch_size - mean_p.size(0), mean_p.size(1)))], dim=0)
            elif mean_p.size(0) > batch_size:
                mean_p = mean_p[:batch_size]

            if attn_p.size(0) < batch_size:
                attn_p = torch.cat([attn_p, attn_p.new_zeros((batch_size - attn_p.size(0), attn_p.size(1)))], dim=0)
            elif attn_p.size(0) > batch_size:
                attn_p = attn_p[:batch_size]

            return torch.cat([mean_p, attn_p], dim=-1)

        pos_pool = _mean_attn_pool(x_dict["pos"], batch_pos, "pos")
        neg_pool = _mean_attn_pool(x_dict["neg"], batch_neg, "neg")
        clause_pool = _mean_attn_pool(x_dict["clause"], batch_clause, "clause")

        g_feat = torch.cat([pos_pool, neg_pool, clause_pool], dim=-1)
        g_feat = _safe_nan_to_num(g_feat, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.use_struct:
            struct = _prepare_struct_x(data, batch_size, self.struct_in_dim, device)
            struct_embed = self.struct_mlp(struct)
            head_in = torch.cat([g_feat, struct_embed], dim=-1)
        else:
            head_in = g_feat

        out = self.predictor(head_in).squeeze(-1)
        out = _safe_nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)
        return out


class HeteroSAGERegressor(HeteroBaseRegressor):
    def __init__(
        self,
        pos_in_dim: int = 7,
        neg_in_dim: int = 8,
        clause_in_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        attn_pool_dropout: float = 0.1,
        attn_lambda_init: float = 0.1,
        use_struct: bool = True,
        struct_in_dim: int = 15,
        gat_heads: int = 4,
        strict_input_dim_check: bool = True,
    ):
        super().__init__(
            conv_type="hetero_sage",
            pos_in_dim=pos_in_dim,
            neg_in_dim=neg_in_dim,
            clause_in_dim=clause_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attn_pool_dropout=attn_pool_dropout,
            attn_lambda_init=attn_lambda_init,
            use_struct=use_struct,
            struct_in_dim=struct_in_dim,
            gat_heads=gat_heads,
            strict_input_dim_check=strict_input_dim_check,
        )


class HeteroGINRegressor(HeteroBaseRegressor):
    def __init__(
        self,
        pos_in_dim: int = 7,
        neg_in_dim: int = 8,
        clause_in_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        attn_pool_dropout: float = 0.1,
        attn_lambda_init: float = 0.1,
        use_struct: bool = True,
        struct_in_dim: int = 15,
        gat_heads: int = 4,
        strict_input_dim_check: bool = True,
    ):
        super().__init__(
            conv_type="hetero_gin",
            pos_in_dim=pos_in_dim,
            neg_in_dim=neg_in_dim,
            clause_in_dim=clause_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attn_pool_dropout=attn_pool_dropout,
            attn_lambda_init=attn_lambda_init,
            use_struct=use_struct,
            struct_in_dim=struct_in_dim,
            gat_heads=gat_heads,
            strict_input_dim_check=strict_input_dim_check,
        )


class HeteroGATRegressor(HeteroBaseRegressor):
    def __init__(
        self,
        pos_in_dim: int = 7,
        neg_in_dim: int = 8,
        clause_in_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        attn_pool_dropout: float = 0.1,
        attn_lambda_init: float = 0.1,
        use_struct: bool = True,
        struct_in_dim: int = 15,
        gat_heads: int = 4,
        strict_input_dim_check: bool = True,
    ):
        super().__init__(
            conv_type="hetero_gat",
            pos_in_dim=pos_in_dim,
            neg_in_dim=neg_in_dim,
            clause_in_dim=clause_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attn_pool_dropout=attn_pool_dropout,
            attn_lambda_init=attn_lambda_init,
            use_struct=use_struct,
            struct_in_dim=struct_in_dim,
            gat_heads=gat_heads,
            strict_input_dim_check=strict_input_dim_check,
        )


def build_model(
    model_type: str,
    pos_in_dim: int = 7,
    neg_in_dim: int = 8,
    clause_in_dim: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.1,
    attn_pool_dropout: float = 0.1,
    attn_lambda_init: float = 0.1,
    use_struct: bool = True,
    struct_in_dim: int = 15,
    gat_heads: int = 4,
    strict_input_dim_check: bool = True,
):
    mt = model_type.lower()
    if mt == "hetero_sage":
        return HeteroSAGERegressor(
            pos_in_dim=pos_in_dim,
            neg_in_dim=neg_in_dim,
            clause_in_dim=clause_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attn_pool_dropout=attn_pool_dropout,
            attn_lambda_init=attn_lambda_init,
            use_struct=use_struct,
            struct_in_dim=struct_in_dim,
            gat_heads=gat_heads,
            strict_input_dim_check=strict_input_dim_check,
        )
    elif mt == "hetero_gin":
        return HeteroGINRegressor(
            pos_in_dim=pos_in_dim,
            neg_in_dim=neg_in_dim,
            clause_in_dim=clause_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attn_pool_dropout=attn_pool_dropout,
            attn_lambda_init=attn_lambda_init,
            use_struct=use_struct,
            struct_in_dim=struct_in_dim,
            gat_heads=gat_heads,
            strict_input_dim_check=strict_input_dim_check,
        )
    elif mt == "hetero_gat":
        return HeteroGATRegressor(
            pos_in_dim=pos_in_dim,
            neg_in_dim=neg_in_dim,
            clause_in_dim=clause_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attn_pool_dropout=attn_pool_dropout,
            attn_lambda_init=attn_lambda_init,
            use_struct=use_struct,
            struct_in_dim=struct_in_dim,
            gat_heads=gat_heads,
            strict_input_dim_check=strict_input_dim_check,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
