from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv


class GroupEmbed(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class PairwiseFeatureEmbedConfig:
    use_distance: bool = False
    distance_dim: int = 32


class PairwiseFeatureEmbed(nn.Module):
    def __init__(self, cfg: PairwiseFeatureEmbedConfig):
        super().__init__()
        self.use_distance = cfg.use_distance
        self.mods = nn.ModuleDict()
        if self.use_distance:
            self.mods["distance"] = GroupEmbed(1, cfg.distance_dim)

    def forward(self, distance: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if not self.use_distance or distance is None:
            return None

        original_shape = distance.shape
        if len(original_shape) == 3:
            distance_flat = distance.reshape(-1, 1)
        else:
            distance_flat = distance

        h = self.mods["distance"](distance_flat)
        if len(original_shape) == 3:
            h = h.reshape(original_shape[0], original_shape[1], -1)
        return h


@dataclass
class NodeFeatureEmbedConfig:
    use_intrinsic: bool = True
    use_normal: bool = True
    use_xyz: bool = False
    intrinsic_dim: int = 32
    normal_dim: int = 32
    xyz_dim: int = 32


class NodeFeatureEmbed(nn.Module):
    def __init__(self, cfg: NodeFeatureEmbedConfig):
        super().__init__()
        self.use_intrinsic = cfg.use_intrinsic
        self.use_normal = cfg.use_normal
        self.use_xyz = cfg.use_xyz

        self.mods = nn.ModuleDict()
        if self.use_intrinsic:
            self.mods["intrinsic"] = GroupEmbed(2, cfg.intrinsic_dim)
        if self.use_normal:
            self.mods["normal"] = GroupEmbed(3, cfg.normal_dim)
        if self.use_xyz:
            self.mods["xyz"] = GroupEmbed(3, cfg.xyz_dim)

        if len(self.mods) == 0:
            raise ValueError("At least one node feature group must be enabled.")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        module_chunks: dict[str, torch.Tensor] = {}
        start_idx = 0
        if self.use_intrinsic:
            module_chunks["intrinsic"] = x[:, start_idx : start_idx + 2]
            start_idx += 2
        if self.use_normal:
            module_chunks["normal"] = x[:, start_idx : start_idx + 3]
            start_idx += 3
        if self.use_xyz:
            module_chunks["xyz"] = x[:, start_idx : start_idx + 3]
            start_idx += 3

        embs = {}
        for k, mod in self.mods.items():
            embs[k] = mod(module_chunks[k])
        return embs


class GeoGNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        *,
        feat_embed_cfg: Optional[NodeFeatureEmbedConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_embed_cfg = feat_embed_cfg
        self.dropout = dropout

        if feat_embed_cfg is None:
            raise ValueError("feat_embed_cfg is required (legacy single-stream path removed).")

        self.feat_embed = NodeFeatureEmbed(feat_embed_cfg)

        in_dim_eff = 0
        if feat_embed_cfg.use_intrinsic:
            in_dim_eff += feat_embed_cfg.intrinsic_dim
        if feat_embed_cfg.use_normal:
            in_dim_eff += feat_embed_cfg.normal_dim
        if feat_embed_cfg.use_xyz:
            in_dim_eff += feat_embed_cfg.xyz_dim

        self.gat1 = GATv2Conv(in_dim_eff, hidden_dim // heads, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)

        self.module_projections = nn.ModuleDict()
        if feat_embed_cfg.use_intrinsic:
            self.module_projections["intrinsic"] = nn.Linear(hidden_dim, feat_embed_cfg.intrinsic_dim)
        if feat_embed_cfg.use_normal:
            self.module_projections["normal"] = nn.Linear(hidden_dim, feat_embed_cfg.normal_dim)
        if feat_embed_cfg.use_xyz:
            self.module_projections["xyz"] = nn.Linear(hidden_dim, feat_embed_cfg.xyz_dim)

        self.module_skip_scale = nn.ParameterDict()
        for k in self.module_projections.keys():
            self.module_skip_scale[k] = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        module_embs = self.feat_embed(x)
        x_concat = torch.cat([module_embs[k] for k in sorted(module_embs.keys())], dim=-1)
        h = self.gat1(x_concat, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)

        result: dict[str, torch.Tensor] = {}
        for k in self.module_projections.keys():
            out = self.module_projections[k](h) + self.module_skip_scale[k] * module_embs[k]
            result[k] = out
        return result


@dataclass
class SinkhornPredictorConfig:
    mlp_hidden: int = 256
    mlp_dropout: float = 0.1
    temp: float = 0.1
    sinkhorn_iters: int = 20
    top_k_sparse: Optional[int] = None
    weight_threshold: Optional[float] = None
    use_distance: bool = False
    sparse_chunk_t: int = 4096
    dist_penalty_lambda: float = 0.0


class PairwiseMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class SinkhornPredictor(nn.Module):
    """
    Sparse-only predictor:
    - Input: per-target candidate source indices (Nt,K)
    - Output: weights w (Nt,K) via softmax over K.
    Dense Sinkhorn path removed.
    """

    def __init__(
        self,
        hidden_dim: int,
        cfg: SinkhornPredictorConfig,
        use_distance: bool = False,
        module_dims: Optional[dict[str, int]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.use_distance = use_distance
        self.module_dims = module_dims or {}

        if use_distance:
            distance_dim = getattr(cfg, "distance_dim", 32)
            pairwise_cfg = PairwiseFeatureEmbedConfig(use_distance=True, distance_dim=distance_dim)
            self.pairwise_embed = PairwiseFeatureEmbed(pairwise_cfg)
        else:
            self.pairwise_embed = None
            distance_dim = 0

        total_module_dim = sum(2 * dim for dim in self.module_dims.values())
        in_dim = total_module_dim + distance_dim
        self.mlp = PairwiseMLP(in_dim, hidden=cfg.mlp_hidden, dropout=cfg.mlp_dropout)

    def forward(
        self,
        h: dict[str, torch.Tensor],
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        *,
        cand_src_idx: torch.Tensor,
        cand_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        device = next(iter(self.parameters())).device

        hs_dict = {k: h[k][source_idx] for k in h.keys()}
        ht_dict = {k: h[k][target_idx] for k in h.keys()}
        first_key = next(iter(hs_dict.keys()))
        device = hs_dict[first_key].device

        Ns, Nt = len(source_idx), len(target_idx)
        cand_src_idx = cand_src_idx.to(device)
        Nt_cand, K = cand_src_idx.shape
        if Nt_cand != Nt:
            raise ValueError(f"cand_src_idx has Nt={Nt_cand} but target_idx has Nt={Nt}")
        if cand_dist is not None and cand_dist.shape != (Nt_cand, K):
            raise ValueError(f"cand_dist must be (Nt,K)={Nt_cand,K}, got {tuple(cand_dist.shape)}")

        hs_c_dict = {mn: hs_dict[mn][cand_src_idx] for mn in hs_dict.keys()}  # (Nt,K,dim)
        ht_dict_full = {mn: ht_dict[mn] for mn in ht_dict.keys()}  # (Nt,dim)

        chunk_t = max(1, int(getattr(cfg, "sparse_chunk_t", 4096)))
        scores = torch.empty((Nt_cand, K), device=device, dtype=torch.float32)

        for t0 in range(0, Nt_cand, chunk_t):
            t1 = min(Nt_cand, t0 + chunk_t)
            bt = t1 - t0
            feat_parts = []

            for mn in sorted(hs_c_dict.keys()):
                hs_c = hs_c_dict[mn][t0:t1]  # (bt,K,dim)
                ht = ht_dict_full[mn][t0:t1]  # (bt,dim)
                ht_rep = ht[:, None, :].expand(bt, K, ht.shape[1])
                feat_parts.append(torch.cat([hs_c.reshape(bt * K, -1), ht_rep.reshape(bt * K, -1)], dim=-1))

            if self.use_distance and self.pairwise_embed is not None and cand_dist is not None:
                d_c = cand_dist[t0:t1].to(device)
                # Stabilize distance magnitude for the MLP
                d_c = torch.log1p(torch.clamp(d_c, min=0.0, max=1e6))
                d_embed = self.pairwise_embed(d_c.unsqueeze(-1))
                feat_parts.append(d_embed.reshape(bt * K, -1))

            feat = torch.cat(feat_parts, dim=1)
            if self.training and feat.requires_grad:
                s_flat = checkpoint(self.mlp, feat, use_reentrant=False)
            else:
                s_flat = self.mlp(feat)
            scores[t0:t1] = s_flat.reshape(bt, K)

        if cand_dist is not None and float(getattr(cfg, "dist_penalty_lambda", 0.0)) > 0.0:
            lam = float(getattr(cfg, "dist_penalty_lambda", 0.0))
            d = cand_dist.to(device=device, dtype=scores.dtype)
            d = torch.nan_to_num(d, nan=0.0, posinf=1e6, neginf=0.0)
            d = torch.clamp(d, min=0.0, max=1e6)
            d = torch.log1p(d)
            scores = scores - lam * (d * d)

        w = torch.softmax(scores / cfg.temp, dim=1)

        if cfg.sinkhorn_iters and cfg.sinkhorn_iters > 0:
            eps = 1e-12
            w = w.to(torch.float64)
            b = torch.ones((Nt,), device=device, dtype=w.dtype)
            a = torch.full((Ns,), float(Nt) / float(Ns), device=device, dtype=w.dtype)

            for _ in range(int(cfg.sinkhorn_iters)):
                col_sum = w.sum(dim=1, keepdim=True)
                col_sum = torch.clamp(col_sum, min=eps)
                w = w * (b[:, None] / col_sum)
                w = torch.clamp(w, min=eps)

                row_sum = torch.zeros((Ns,), device=device, dtype=w.dtype)
                row_sum.scatter_add_(0, cand_src_idx.reshape(-1), w.reshape(-1))
                scale = a / torch.clamp(row_sum, min=eps)
                w = w * scale[cand_src_idx]
                w = torch.clamp(w, min=eps)

            w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=eps)
            w = w.to(torch.float32)

        if not self.training:
            if cfg.top_k_sparse is not None and cfg.top_k_sparse < K:
                topk_vals, topk_idx = torch.topk(w, k=cfg.top_k_sparse, dim=1)
                w_sparse = torch.zeros_like(w)
                w_sparse.scatter_(1, topk_idx, topk_vals)
                w = w_sparse / (w_sparse.sum(dim=1, keepdim=True) + 1e-8)
            elif cfg.weight_threshold is not None:
                w = w * (w >= cfg.weight_threshold)
                w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

        return w


class GeoGNNOT(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        hidden_dim: int = 128,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        sinkhorn_cfg: Optional[SinkhornPredictorConfig] = None,
        *,
        feat_embed_cfg: Optional[NodeFeatureEmbedConfig] = None,
        use_distance: bool = False,
    ) -> None:
        super().__init__()
        if feat_embed_cfg is None:
            raise ValueError("feat_embed_cfg is required")

        self.encoder = GeoGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            heads=gat_heads,
            dropout=gat_dropout,
            feat_embed_cfg=feat_embed_cfg,
        )

        self.sinkhorn_cfg = sinkhorn_cfg or SinkhornPredictorConfig()
        module_dims: dict[str, int] = {}
        if feat_embed_cfg.use_intrinsic:
            module_dims["intrinsic"] = feat_embed_cfg.intrinsic_dim
        if feat_embed_cfg.use_normal:
            module_dims["normal"] = feat_embed_cfg.normal_dim
        if feat_embed_cfg.use_xyz:
            module_dims["xyz"] = feat_embed_cfg.xyz_dim

        self.predictor = SinkhornPredictor(
            hidden_dim=hidden_dim,
            cfg=self.sinkhorn_cfg,
            use_distance=use_distance,
            module_dims=module_dims,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        tuning: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        *,
        cand_src_idx: torch.Tensor,
        cand_dist: Optional[torch.Tensor] = None,
        return_W: bool = True,
    ) -> dict:
        # Defensive: some older checkpoints/eval code paths may accidentally add a leading singleton dim
        # (e.g. tuning shaped (1, N, D)). The model always expects per-node tuning shaped (N, D).
        if tuning.dim() == 3 and tuning.size(0) == 1:
            tuning = tuning.squeeze(0)

        h = self.encoder(x, edge_index)

        W = self.predictor(
            h=h,
            source_idx=source_idx,
            target_idx=target_idx,
            cand_src_idx=cand_src_idx,
            cand_dist=cand_dist,
        )

        if not return_W:
            return {"W": W}

        tuning_src = tuning[source_idx]
        tuning_pred = (W.to(tuning.dtype).unsqueeze(-1) * tuning_src[cand_src_idx]).sum(dim=1)
        return {"W": W, "tuning_pred": tuning_pred}


class TopologicalRegularizationLoss(nn.Module):
    """
    Penalizes scattered source connections for a single target.
    
    If W[t, :] has high weights on candidate sources i and k,
    and dist(i, k) is large, this loss increases.
    
    Efficient implementation approximates this by minimizing the 
    weighted variance of source coordinates (or embeddings) for each target.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        W: torch.Tensor,
        cand_src_idx: torch.Tensor,
        source_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            W: (Nt, K) Soft assignments (sum_k W[t,k] = 1)
            cand_src_idx: (Nt, K) Indices of candidates in source_coords
            source_coords: (Ns, D) Coordinates (e.g., Sphere or MDS)
        """
        Nt, K = W.shape
        src_pos = source_coords[cand_src_idx]
        center = (W.unsqueeze(-1) * src_pos).sum(dim=1)
        diff = src_pos - center.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)
        loss = (W * dist_sq).sum() / Nt
        return loss

