from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from torch_geometric.data import Data


@dataclass(frozen=True)
class SubjectSpec:
    subject: str  # e.g., R1, S1..S6
    hemi: str  # lh/rh


def _load_npy(p: Path) -> np.ndarray:
    return np.load(p, allow_pickle=True)


def build_geodesic_csr(n: int, edge_index: np.ndarray, dist: np.ndarray) -> csr_matrix:
    """
    Build sparse distance matrix D (n,n) from (2,K) indices and (K,) distances.
    Symmetrize by adding reversed edges.
    """
    i = edge_index[0].astype(np.int64)
    j = edge_index[1].astype(np.int64)
    d = dist.astype(np.float32)

    ii = np.concatenate([i, j])
    jj = np.concatenate([j, i])
    dd = np.concatenate([d, d]).astype(np.float32)
    return coo_matrix((dd, (ii, jj)), shape=(n, n), dtype=np.float32).tocsr()


@dataclass
class PreprocessedSubject:
    spec: SubjectSpec
    root: Path  # repo root
    use_distances: bool = True

    def subject_dir(self) -> Path:
        return self.root / "preprocessing" / "outputs" / self.spec.subject / self.spec.hemi

    def load(self) -> Tuple[Data, dict]:
        d = self.subject_dir()
        x = _load_npy(d / "node_features.npy").astype(np.float32)
        edge_index = _load_npy(d / "edge_index.npy").astype(np.int64)
        mid_coords = _load_npy(d / "mid_coords.npy").astype(np.float32)
        area = _load_npy(d / "area_labels.npy").astype(np.int64)
        tuning = _load_npy(d / "tuning.npy").astype(np.float32)
        has_tuning = _load_npy(d / "has_tuning.npy").astype(bool)
        global_vertex_idx = _load_npy(d / "global_vertex_idx.npy").astype(np.int64)
        faces_sub = _load_npy(d / "faces_sub.npy").astype(np.int64)

        geo_csr: Optional[csr_matrix] = None
        if self.use_distances:
            geo_csr_cache = d / "geodesic_csr.npz"
            if geo_csr_cache.exists():
                # Fast path: load pre-cached CSR matrix
                geo_csr = load_npz(geo_csr_cache)
            else:
                # Slow path: build from raw edge data and cache for next time
                gei_p = d / "geodesic_edge_index.npy"
                gd_p = d / "geodesic_dist.npy"
                if gei_p.exists() and gd_p.exists():
                    gei = _load_npy(gei_p).astype(np.int64)
                    gd = _load_npy(gd_p).astype(np.float32)
                    geo_csr = build_geodesic_csr(x.shape[0], gei, gd)
                    # Save cache for faster subsequent loads
                    try:
                        save_npz(geo_csr_cache, geo_csr)
                    except Exception:
                        pass  # Ignore write errors (e.g., read-only filesystem)

        # IMPORTANT: only use V1 nodes with valid tuning as sources.
        source_mask = (area == 1) & has_tuning
        target_mask = (area == 2) | (area == 3) | (area == 4)
        target_valid_mask = target_mask & has_tuning

        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            mid_coords=torch.from_numpy(mid_coords),
            area=torch.from_numpy(area),
            tuning=torch.from_numpy(tuning),
            has_tuning=torch.from_numpy(has_tuning.astype(np.bool_)),
            global_vertex_idx=torch.from_numpy(global_vertex_idx),
            faces_sub=torch.from_numpy(faces_sub),
        )

        meta = {
            "subject": self.spec.subject,
            "hemi": self.spec.hemi,
            "source_idx": np.where(source_mask)[0].astype(np.int64),
            "target_idx": np.where(target_mask)[0].astype(np.int64),
            "target_valid_idx": np.where(target_valid_mask)[0].astype(np.int64),
            "geo_csr": geo_csr,
        }
        return data, meta


@dataclass(frozen=True)
class FeatureNormStats:
    intrinsic_mu: np.ndarray  # (2,)
    intrinsic_std: np.ndarray  # (2,)
    xyz_mu: np.ndarray  # (3,)
    xyz_std: np.ndarray  # (3,)
    eps: float = 1e-8


def compute_feature_norm_stats(
    node_features: np.ndarray,
    mid_coords: np.ndarray,
    *,
    eps: float = 1e-8,
) -> FeatureNormStats:
    if node_features.shape[1] < 5:
        raise ValueError(f"node_features must have at least 5 dims, got {node_features.shape}")
    if mid_coords.shape[1] != 3:
        raise ValueError(f"mid_coords must be (N,3), got {mid_coords.shape}")

    intrinsic = node_features[:, 0:2].astype(np.float32)
    xyz = mid_coords.astype(np.float32)

    intrinsic_mu = intrinsic.mean(axis=0)
    intrinsic_std = intrinsic.std(axis=0) + eps
    xyz_mu = xyz.mean(axis=0)
    xyz_std = xyz.std(axis=0) + eps

    return FeatureNormStats(
        intrinsic_mu=intrinsic_mu,
        intrinsic_std=intrinsic_std,
        xyz_mu=xyz_mu,
        xyz_std=xyz_std,
        eps=eps,
    )


def apply_feature_norm_numpy(
    node_features: np.ndarray,
    mid_coords: np.ndarray,
    stats: FeatureNormStats,
    *,
    unit_normal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    x = node_features.astype(np.float32).copy()
    c = mid_coords.astype(np.float32).copy()

    x[:, 0:2] = (x[:, 0:2] - stats.intrinsic_mu[None, :]) / stats.intrinsic_std[None, :]

    if unit_normal:
        n = x[:, 2:5]
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        x[:, 2:5] = n / (n_norm + stats.eps)

    c = (c - stats.xyz_mu[None, :]) / stats.xyz_std[None, :]
    return x, c


def compute_vertex_normals_numpy(coords: np.ndarray, faces: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    v = coords.astype(np.float32)
    f = faces.astype(np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {faces.shape}")

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # (F,3)

    n = np.zeros_like(v, dtype=np.float32)
    np.add.at(n, f[:, 0], fn)
    np.add.at(n, f[:, 1], fn)
    np.add.at(n, f[:, 2], fn)

    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (n_norm + eps)


def geo_csr_column_topk_candidates(
    geo_csr: csr_matrix,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    *,
    k: int,
    fill_random: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if geo_csr.shape[0] != geo_csr.shape[1]:
        raise ValueError(f"geo_csr must be square, got {geo_csr.shape}")

    n = geo_csr.shape[0]
    source_idx = source_idx.astype(np.int64)
    target_idx = target_idx.astype(np.int64)
    Ns = source_idx.shape[0]
    Nt = target_idx.shape[0]
    K = min(k, Ns)

    pos = np.full((n,), -1, dtype=np.int64)
    pos[source_idx] = np.arange(Ns, dtype=np.int64)

    geo_csc = geo_csr.tocsc(copy=False)

    cand_src_pos = np.empty((Nt, K), dtype=np.int64)
    cand_dist = np.empty((Nt, K), dtype=np.float32)

    rng = np.random.default_rng(seed)
    for ti, t_global in enumerate(target_idx):
        col_start = geo_csc.indptr[t_global]
        col_end = geo_csc.indptr[t_global + 1]
        rows = geo_csc.indices[col_start:col_end]
        d = geo_csc.data[col_start:col_end].astype(np.float32, copy=False)

        src_pos = pos[rows]
        m = src_pos >= 0
        if not np.any(m):
            if fill_random:
                cand_src_pos[ti] = rng.choice(Ns, size=K, replace=(K > Ns))
                cand_dist[ti] = np.inf
            else:
                raise ValueError(f"No source-neighbors in geo_csr for target node {t_global}")
            continue

        src_pos = src_pos[m]
        d = d[m]

        if src_pos.shape[0] <= K:
            order = np.argsort(d)
            sel_pos = src_pos[order]
            sel_d = d[order]
            if sel_pos.shape[0] < K and fill_random:
                pad = rng.choice(Ns, size=K - sel_pos.shape[0], replace=True)
                sel_pos = np.concatenate([sel_pos, pad])
                sel_d = np.concatenate([sel_d, np.full((K - sel_d.shape[0],), np.inf, dtype=np.float32)])
        else:
            idx = np.argpartition(d, kth=K - 1)[:K]
            idx = idx[np.argsort(d[idx])]
            sel_pos = src_pos[idx]
            sel_d = d[idx]

        cand_src_pos[ti] = sel_pos[:K]
        cand_dist[ti] = sel_d[:K]

    return cand_src_pos, cand_dist


def geo_csr_gather_candidate_distances(
    geo_csr: csr_matrix,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    cand_src_pos: np.ndarray,
    *,
    missing_to_inf: bool = True,
) -> np.ndarray:
    if cand_src_pos.ndim != 2:
        raise ValueError(f"cand_src_pos must be (Nt,K), got {cand_src_pos.shape}")
    source_idx = source_idx.astype(np.int64)
    target_idx = target_idx.astype(np.int64)
    Nt, K = cand_src_pos.shape
    if target_idx.shape[0] != Nt:
        raise ValueError(f"target_idx length must match cand_src_pos Nt={Nt}, got {target_idx.shape}")

    src_global = source_idx[cand_src_pos.astype(np.int64, copy=False)]  # (Nt,K)
    tgt_global = target_idx[:, None].repeat(K, axis=1)  # (Nt,K)

    rows = src_global.reshape(-1)
    cols = tgt_global.reshape(-1)

    vals = np.asarray(geo_csr[rows, cols]).reshape(-1).astype(np.float32, copy=False)
    if missing_to_inf:
        missing = (vals == 0.0) & (rows != cols)
        if np.any(missing):
            vals = vals.copy()
            vals[missing] = np.inf

    return vals.reshape(Nt, K)

