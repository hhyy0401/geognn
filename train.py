from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from geognn_ot.alignment import align_with_node_correspondence


def _normalize_xy(coords: np.ndarray) -> np.ndarray:
    """Normalize 2D coords to [0,1]x[0,1] (baseline-style)."""
    c = np.asarray(coords, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError(f"coords must be (N,2+) got {c.shape}")
    x = c[:, 0]
    y = c[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = max(1e-8, xmax - xmin)
    dy = max(1e-8, ymax - ymin)
    out = c.copy()
    out[:, 0] = (x - xmin) / dx
    out[:, 1] = (y - ymin) / dy
    return out


def _pca_align_xy(coords: np.ndarray) -> np.ndarray:
    """Rotate 2D coords by PCA so the first principal axis aligns with +x (baseline-style alignment)."""
    c = np.asarray(coords, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] < 2:
        raise ValueError(f"coords must be (N,2+) got {c.shape}")
    xy = c[:, :2].copy()
    mu = xy.mean(axis=0, keepdims=True)
    xyc = xy - mu
    # SVD for principal directions
    _, _, vt = np.linalg.svd(xyc, full_matrices=False)
    R = vt[:2, :]  # 2x2
    aligned = xyc @ R.T
    # deterministic sign: make x-axis point to larger max extent
    if np.max(aligned[:, 0]) < -np.min(aligned[:, 0]):
        aligned[:, 0] *= -1
    if np.max(aligned[:, 1]) < -np.min(aligned[:, 1]):
        aligned[:, 1] *= -1
    out = aligned + mu  # keep same centroid
    return out


def _rotate_to_align_x_by_area_centroids(xy: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Baseline_new-style PCA alignment: rotate so principal axis of area centroids aligns with +x."""
    P = np.asarray(xy, dtype=float)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError(f"xy must be (N,2+), got {P.shape}")
    a = np.asarray(areas, dtype=int).reshape(-1)
    if a.shape[0] != P.shape[0]:
        raise ValueError(f"areas length {a.shape[0]} != coords length {P.shape[0]}")

    centroids: list[list[float]] = []
    for area_id in np.unique(a):
        m = a == int(area_id)
        if not np.any(m):
            continue
        centroids.append([float(P[m, 0].mean()), float(P[m, 1].mean())])
    if len(centroids) < 2:
        return P[:, :2]

    C = np.asarray(centroids, dtype=float)
    Cc = C - C.mean(axis=0, keepdims=True)
    cov = np.cov(Cc.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    ang = float(np.arctan2(principal[1], principal[0]))
    c, s = float(np.cos(-ang)), float(np.sin(-ang))
    R = np.asarray([[c, -s], [s, c]], dtype=float)
    return (P[:, :2] @ R.T).astype(float, copy=False)


def _align_mds_like_baseline(
    xy: np.ndarray,
    areas: np.ndarray,
    *,
    center_idx: int | None = None,
) -> np.ndarray:
    """Baseline_new MDS alignment.

    Steps:
    - rotate by PCA of area centroids
    - rotate so V1 centroid points left
    - translate to is_center node when available; otherwise mean-center
    """
    P = _rotate_to_align_x_by_area_centroids(xy, areas)
    a = np.asarray(areas, dtype=int).reshape(-1)

    # Rotate so V1 centroid points left
    m1 = a == 1
    if np.any(m1):
        cx = float(P[m1, 0].mean())
        cy = float(P[m1, 1].mean())
        cur = float(np.arctan2(cy, cx))
        delta = float(np.pi - cur)
        c, s = float(np.cos(delta)), float(np.sin(delta))
        R = np.asarray([[c, -s], [s, c]], dtype=float)
        P = (P @ R.T).astype(float, copy=False)

    # Center (baseline uses an is_center node; fall back to global mean)
    if center_idx is not None and 0 <= int(center_idx) < int(P.shape[0]):
        c = P[int(center_idx)]
    else:
        c = P.mean(axis=0, keepdims=True).reshape(-1)
    P = P - c[None, :]
    return P


def _load_subject_pkl(repo_root: Path, subject: str, hemi: str) -> dict:
    p = Path(repo_root) / "data" / f"{subject}_{hemi}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Missing PKL: {p}")
    import pickle

    with open(p, "rb") as f:
        return pickle.load(f)


def _pkl_lookup(d: dict, key: int):
    # PKL keys sometimes stored as int or str
    if key in d:
        return d[key]
    sk = str(int(key))
    if sk in d:
        return d[sk]
    return None


def _coords_from_pkl_loc(pkl_dict: dict, global_vertex_idx: np.ndarray) -> np.ndarray:
    g = np.asarray(global_vertex_idx, dtype=np.int64).reshape(-1)
    out = np.zeros((g.shape[0], 2), dtype=np.float32)
    for i, gid in enumerate(g):
        v = _pkl_lookup(pkl_dict, int(gid))
        loc = v.get("loc", None) if isinstance(v, dict) else None
        if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
            # Use only first two dimensions (x, y) for 2D plotting
            out[i, 0] = float(loc[0])
            out[i, 1] = float(loc[1])
        else:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
    return out


def _tuning_from_pkl(pkl_dict: dict, global_vertex_idx: np.ndarray) -> np.ndarray:
    g = np.asarray(global_vertex_idx, dtype=np.int64).reshape(-1)
    out = np.zeros((g.shape[0], 2), dtype=np.float32)
    for i, gid in enumerate(g):
        v = _pkl_lookup(pkl_dict, int(gid))
        tun = v.get("tuning", None) if isinstance(v, dict) else None
        if isinstance(tun, (list, tuple)) and len(tun) >= 2:
            out[i, 0] = float(tun[0])
            out[i, 1] = float(tun[1])
        else:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
    return out


def _is_center_from_pkl(pkl_dict: dict, global_vertex_idx: np.ndarray) -> np.ndarray:
    g = np.asarray(global_vertex_idx, dtype=np.int64).reshape(-1)
    out = np.zeros((g.shape[0],), dtype=bool)
    for i, gid in enumerate(g):
        v = _pkl_lookup(pkl_dict, int(gid))
        flag = v.get("is_center", 0) if isinstance(v, dict) else 0
        out[i] = bool(int(flag) == 1)
    return out

def _normalize_tuning(tuning_xy: np.ndarray, tmin: np.ndarray | None, tmax: np.ndarray | None, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize 2D tuning to [0,1] per-dimension, matching training normalization."""
    t = np.asarray(tuning_xy, dtype=np.float32)
    if t.ndim != 2 or t.shape[1] != 2:
        raise ValueError(f"tuning must be (N,2) got {t.shape}")
    if tmin is None or tmax is None:
        # fallback: normalize using data min/max
        tmin = np.min(t, axis=0)
        tmax = np.max(t, axis=0)
    denom = np.maximum(eps, (tmax - tmin))
    tn = (t - tmin[None, :]) / denom[None, :]
    return np.clip(tn, 0.0, 1.0)

def _tuning_to_rgb(tuning_xy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Map 2D tuning to RGB via HSV hue=angle. Deterministic + bounded."""
    t = np.asarray(tuning_xy, dtype=np.float32)
    if t.ndim != 2 or t.shape[1] != 2:
        raise ValueError(f"tuning must be (N,2) got {t.shape}")
    ang = np.arctan2(t[:, 1], t[:, 0])  # [-pi,pi]
    h = (ang + np.pi) / (2 * np.pi)     # [0,1]
    s = np.ones_like(h)
    v = np.ones_like(h)
    # HSV->RGB
    i = np.floor(h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    r = v * (1 - (1 - f) * s)
    i = i % 6
    rgb = np.zeros((t.shape[0], 3), dtype=np.float32)
    masks = [
        (i == 0), (i == 1), (i == 2), (i == 3), (i == 4), (i == 5)
    ]
    rgb[masks[0]] = np.stack([v[masks[0]], r[masks[0]], p[masks[0]]], axis=1)
    rgb[masks[1]] = np.stack([q[masks[1]], v[masks[1]], p[masks[1]]], axis=1)
    rgb[masks[2]] = np.stack([p[masks[2]], v[masks[2]], r[masks[2]]], axis=1)
    rgb[masks[3]] = np.stack([p[masks[3]], q[masks[3]], v[masks[3]]], axis=1)
    rgb[masks[4]] = np.stack([r[masks[4]], p[masks[4]], v[masks[4]]], axis=1)
    rgb[masks[5]] = np.stack([v[masks[5]], p[masks[5]], q[masks[5]]], axis=1)
    return rgb.clip(0, 1)

def _save_align_html_png(
    out_html: Path,
    out_png: Path,
    coords_3d: np.ndarray,
    area_labels: np.ndarray,
    colors: np.ndarray,
    *,
    title: str,
    template_coords_3d: np.ndarray | None = None,
    template_area_labels: np.ndarray | None = None,
    template_name: str = "Brain A (template)",
    subject_name: str = "Brain B (subject)",
) -> None:
    """Interactive Plotly HTML of 3D coordinates colored by visual area.

    Overlay two brains when template_* is provided:
      - Brain A (template): darker shade
      - Brain B (subject):  lighter shade
    Same visual area keeps hue; only lightness differs.
    """

    import plotly.graph_objects as go
    import plotly.io as pio

    def _as_xyz(x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] < 3:
            raise ValueError(f"coords_3d must be (N,3+); got {x.shape}")
        return x

    def _as_area(a0, n):
        a0 = np.asarray(a0, dtype=int).reshape(-1)
        if a0.shape[0] != n:
            raise ValueError(f"area_labels length {a0.shape[0]} != coords length {n}")
        return a0

    xyz = _as_xyz(coords_3d)
    a = _as_area(area_labels, xyz.shape[0])

    from TUNING_COLOR_UTILS import get_tuning_colormap, round_color_bins

    def _scale_rgb(rgb: tuple[int, int, int], factor: float) -> str:
        r, g, b = rgb
        rr = int(max(0, min(255, round(r * factor))))
        gg = int(max(0, min(255, round(g * factor))))
        bb = int(max(0, min(255, round(b * factor))))
        return f"rgb({rr}, {gg}, {bb})"

    base_rgb = {
        1: (255, 128, 0),    # V1 orange
        2: (153, 0, 204),    # V2 purple
        3: (204, 102, 204),  # V3 light purple
        4: (0, 204, 0),      # V4 green
    }
    area_colors_A = {k: _scale_rgb(v, 0.85) for k, v in base_rgb.items()}  # template darker
    area_colors_B = {k: _scale_rgb(v, 1.15) for k, v in base_rgb.items()}  # subject lighter
    area_names = {1: "V1", 2: "V2", 3: "V3", 4: "V4"}

    fig = go.Figure()

    def _add_brain(xyz_b: np.ndarray, a_b: np.ndarray, *, brain_label: str, colors_by_area: dict[int, str]):
        for area_id in (1, 2, 3, 4):
            m = a_b == area_id
            if not np.any(m):
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=xyz_b[m, 0],
                    y=xyz_b[m, 1],
                    z=xyz_b[m, 2],
                    mode="markers",
                    marker=dict(size=2, color=colors_by_area.get(area_id, "rgb(128, 128, 128)"), opacity=0.85),
                    name=f"{brain_label} {area_names.get(area_id, f'Area {area_id}')}",
                    legendgroup=brain_label,
                )
            )

    if template_coords_3d is not None and template_area_labels is not None:
        xyzA = _as_xyz(template_coords_3d)
        aA = _as_area(template_area_labels, xyzA.shape[0])
        _add_brain(xyzA, aA, brain_label=template_name, colors_by_area=area_colors_A)

    _add_brain(xyz, a, brain_label=subject_name, colors_by_area=area_colors_B)

    fig.update_layout(
        title=title,
        showlegend=True,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        width=900,
        height=800,
    )

    pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")

    # Best-effort PNG screenshot
    try:
        pio.write_image(fig, str(out_png), format="png", scale=2)
    except Exception as e:
        print(f"[warn] could not write plotly PNG (kaleido missing?): {out_png} ({e})")

def _save_tuning_compare_png(
    out_png: Path,
    coords_xy: np.ndarray,
    tuning_true_all: np.ndarray,
    tuning_pred_all: np.ndarray,
    *,
    v1_mask: np.ndarray,
    tag: str,
    title_left: str = "True",
    title_right: str = "Pred",
    data_name: str | None = None,
) -> None:
    """Three-panel tuning comparison using v2 9-color paper palette."""

    import matplotlib.pyplot as plt
    from TUNING_COLOR_UTILS import (
        compute_tuning_colors_v2 as compute_tuning_colors,
        compute_tuning_colors_r_v2 as compute_tuning_colors_r,
        get_v2_colormap as get_tuning_colormap,
        should_flip_y_red_bottom,
    )

    xy = np.asarray(coords_xy, dtype=float)[:, :2]
    v1_mask_b = np.asarray(v1_mask, dtype=bool)

    # 1. Compute True Colors (anchored to V1 True)
    true_c = np.asarray(
        compute_tuning_colors(tuning_true_all, v1_mask=v1_mask_b, tag=tag),
        dtype=float,
    )

    # 2. Compute Predicted Colors (anchored to V1 True to match baseline convention)
    tuning_pred_fixed = tuning_pred_all.copy()
    if np.any(v1_mask_b):
        tuning_pred_fixed[v1_mask_b] = tuning_true_all[v1_mask_b]
    
    pred_c = np.asarray(
        compute_tuning_colors(tuning_pred_fixed, v1_mask=v1_mask_b, tag=tag),
        dtype=float,
    )

    # 3. Handle Y-flip (Enforce Red/Low-index at bottom)
    import re
    data_name_str = str(data_name) if data_name else ""
    is_rotated = bool(re.search(r'_(90|180|270)(_|$)', data_name_str))
    
    # Use discrete colors for flip check (matches baseline_new behavior)
    true_c_discrete = np.round(true_c * 10.0) / 10.0
    true_c_discrete = np.clip(true_c_discrete, 0.0, 1.0)
    
    if is_rotated:
        # For rotated datasets, use fixed flip based on hemisphere tag
        if tag == "lh":
            flip_y = False
        elif tag == "rh":
            flip_y = True
        else:
            flip_y = False
    else:
        try:
            mask_for_flip = v1_mask_b if np.any(v1_mask_b) else np.ones(len(xy), dtype=bool)
            flip_y = should_flip_y_red_bottom(xy[mask_for_flip], true_c_discrete[mask_for_flip])
        except Exception:
            flip_y = False

    if flip_y:
        xy = xy.copy()
        xy[:, 1] *= -1.0

    # 4. Colormap and RGBA computation
    cmap = get_tuning_colormap()
    # v2 colormap has 10 colors: 0.0, 0.1, ..., 0.9
    true_rgba = [cmap(float(c)) for c in np.clip(true_c, 0.0, 0.95)]
    polar_rgba = [cmap(float(c)) for c in np.clip(pred_c, 0.0, 0.95)]

    # 5. Eccentricity (radius-based) colors
    # Always using the fixed V1-true anchored version for prediction.
    pred_r = np.asarray(
        compute_tuning_colors_r(tuning_pred_fixed, v1_mask=v1_mask_b, tag=tag),
        dtype=float,
    )
    ecc_rgba = [cmap(float(c)) for c in np.clip(pred_r, 0.0, 0.95)]

    # 6. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].scatter(xy[:, 0], xy[:, 1], s=10, c=true_rgba)
    axes[1].scatter(xy[:, 0], xy[:, 1], s=10, c=polar_rgba)
    axes[2].scatter(xy[:, 0], xy[:, 1], s=10, c=ecc_rgba)

    axes[0].set_title(title_left)
    axes[1].set_title("Polar angle")
    axes[2].set_title("Eccentricity")
    for ax in axes:
        ax.set_aspect("equal", "box")
        ax.axis("off")

    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.0)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
from geognn_ot.dataset import (
    FeatureNormStats,
    PreprocessedSubject,
    SubjectSpec,
    apply_feature_norm_numpy,
    compute_feature_norm_stats,
    compute_vertex_normals_numpy,
    geo_csr_column_topk_candidates,
    geo_csr_gather_candidate_distances,
)
from geognn_ot.model import GeoGNNOT, NodeFeatureEmbedConfig, SinkhornPredictorConfig, TopologicalRegularizationLoss


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--repo-root", type=str, required=True)
    train_parser.add_argument("--hemi", type=str, choices=["lh", "rh"], required=True)
    train_parser.add_argument("--train-subjects", type=str, nargs="+", default=["R1"])
    train_parser.add_argument("--out-dir", type=str, required=True)
    train_parser.add_argument("--ckpt-out", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=2000)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--scheduler", type=str, choices=["cosine", "step"], default="cosine")
    train_parser.add_argument("--clip-norm", type=float, default=1.0)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--gat-heads", type=int, default=4)
    train_parser.add_argument("--gat-dropout", type=float, default=0.0)
    train_parser.add_argument("--temp", type=float, default=0.3)
    train_parser.add_argument("--sinkhorn-iters", type=int, default=20)
    train_parser.add_argument("--dist-penalty-lambda", type=float, default=0.0)
    train_parser.add_argument("--topo-lambda", type=float, default=0.0, help="Strength of topological regularization loss.")
    train_parser.add_argument("--grid-candidates-out", type=str, default=None, help="Save successful hparams to this JSON for HPO.")
    train_parser.add_argument("--mlp-hidden", type=int, default=256)
    train_parser.add_argument("--mlp-dropout", type=float, default=0.0)
    train_parser.add_argument("--cand-k-src", type=int, default=64)
    train_parser.add_argument("--cand-method", type=str, choices=["score_topk"], default="score_topk")
    train_parser.add_argument("--cand-k-rand", type=int, default=0, help="Number of extra random source candidates per target (if implemented).")
    train_parser.add_argument("--sparse-chunk-t", type=int, default=4096)
    train_parser.add_argument("--cand-refresh-every", type=int, default=1)
    train_parser.add_argument("--score-topk-src-chunk", type=int, default=512)
    train_parser.add_argument("--use-intrinsic", action="store_true")
    train_parser.add_argument("--use-normal", action="store_true")
    train_parser.add_argument("--use-xyz", action="store_true")
    train_parser.add_argument("--use-distance", action="store_true")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--max-train-mse", type=float, default=None)
    train_parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 to disable).")
    train_parser.add_argument("--log-every", type=int, default=50)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--repo-root", type=str, required=True)
    eval_parser.add_argument("--ckpt", type=str, required=True)
    eval_parser.add_argument("--subjects", type=str, nargs="+", default=["R1", "S1", "S2", "S3", "S4", "S5", "S6"])
    eval_parser.add_argument("--hemis", type=str, nargs="+", choices=["lh", "rh"], default=["lh", "rh"])
    eval_parser.add_argument("--out-dir", type=str, required=True)
    eval_parser.add_argument("--device", type=str, default="cpu")
    eval_parser.add_argument("--cand-method", type=str, choices=["score_topk"], default="score_topk")
    eval_parser.add_argument("--cand-k-src", type=int, default=64)
    eval_parser.add_argument("--cand-k-rand", type=int, default=0)
    eval_parser.add_argument("--sparse-chunk-t", type=int, default=4096)
    eval_parser.add_argument("--score-topk-src-chunk", type=int, default=512)
    eval_parser.add_argument("--template-subject", type=str, default="R1")
    eval_parser.add_argument("--use-intrinsic", action="store_true")
    eval_parser.add_argument("--use-normal", action="store_true")
    eval_parser.add_argument("--use-xyz", action="store_true")
    eval_parser.add_argument("--use-distance", action="store_true")
    eval_parser.add_argument("--dist-penalty-lambda", type=float, default=None)

    return parser.parse_args()


def get_module_name(use_intrinsic: bool, use_normal: bool, use_xyz: bool, use_distance: bool) -> str:
    parts = []
    if use_intrinsic:
        parts.append("intrinsic")
    if use_normal:
        parts.append("normal")
    if use_xyz:
        parts.append("xyz")
    if use_distance:
        parts.append("distance")
    return "_".join(parts) if parts else "none"


def cmd_train(args):
    repo_root = Path(args.repo_root).resolve()
    device = torch.device(args.device)
    # Reproducibility
    seed = int(getattr(args, "seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if len(args.train_subjects) != 1:
        raise ValueError("train currently supports exactly one subject in --train-subjects")

    module_name = get_module_name(args.use_intrinsic, args.use_normal, args.use_xyz, args.use_distance)

    spec = SubjectSpec(subject=args.train_subjects[0], hemi=args.hemi)
    need_geo = (
        args.use_distance
        or float(getattr(args, "dist_penalty_lambda", 0.0)) > 0.0
    )
    data_loader = PreprocessedSubject(spec=spec, root=repo_root, use_distances=need_geo)
    data, meta = data_loader.load()

    source_idx = torch.from_numpy(meta["source_idx"]).to(device)
    target_idx = torch.from_numpy(meta["target_valid_idx"]).to(device)
    edge_index = data.edge_index.to(device)
    tuning = data.tuning.to(device)

    # Feature/coord normalization stats from source/template (this subject)
    x_raw = data.x.cpu().numpy()
    coords_raw = data.mid_coords.cpu().numpy()
    feat_norm = compute_feature_norm_stats(x_raw, coords_raw, eps=1e-8)
    x_np, coords_np = apply_feature_norm_numpy(x_raw, coords_raw, feat_norm, unit_normal=True)
    x_norm = torch.from_numpy(x_np).to(device)
    coords = torch.from_numpy(coords_np).to(device)

    x_parts = []
    if args.use_intrinsic:
        x_parts.append(x_norm[:, 0:2])
    if args.use_normal:
        x_parts.append(x_norm[:, 2:5])
    if args.use_xyz:
        x_parts.append(coords)
    if not x_parts:
        raise ValueError("Enable at least one of --use-intrinsic/--use-normal/--use-xyz")
    x = torch.cat(x_parts, dim=-1)

    # Normalize tuning using source stats only (Standardized to Z-score)
    src_tuning = tuning[source_idx]
    tuning_mu = src_tuning.mean(dim=0)
    tuning_std = src_tuning.std(dim=0).clamp_min(1e-8)
    tuning_min = src_tuning.min(dim=0).values
    tuning_max = src_tuning.max(dim=0).values

    tuning_normalized = (tuning - tuning_mu[None, :]) / tuning_std[None, :]

    tuning_stats = {
        "norm_type": "zscore",
        "mu": tuning_mu.detach().cpu().numpy().tolist(),
        "std": tuning_std.detach().cpu().numpy().tolist(),
        "min": tuning_min.detach().cpu().numpy().tolist(),
        "max": tuning_max.detach().cpu().numpy().tolist(),
        "eps": 1e-8,
    }

    cand_src_idx = None
    cand_dist = None
    needs_dist_values = args.use_distance or float(getattr(args, "dist_penalty_lambda", 0.0)) > 0.0
    if needs_dist_values and meta["geo_csr"] is None:
        raise ValueError("use_distance=True requires geodesic distances (geo_csr).")

    topo_loss_fn = None
    source_coords_for_topo = None
    if float(getattr(args, "topo_lambda", 0.0)) > 0.0:
        topo_loss_fn = TopologicalRegularizationLoss()
        source_coords_for_topo = coords

    if args.cand_method == "score_topk":
        pass
    else:
        raise ValueError(f"Unknown cand_method: {args.cand_method}")

    if cand_src_idx is not None and cand_src_idx.shape[0] != int(target_idx.numel()):
        raise ValueError(f"cand_src_idx Nt mismatch: {tuple(cand_src_idx.shape)} vs Nt={int(target_idx.numel())}")

    feat_embed_cfg = NodeFeatureEmbedConfig(
        use_intrinsic=args.use_intrinsic,
        use_normal=args.use_normal,
        use_xyz=args.use_xyz,
        intrinsic_dim=32,
        normal_dim=32,
        xyz_dim=32,
    )

    sinkhorn_cfg = SinkhornPredictorConfig(
        mlp_hidden=args.mlp_hidden,
        mlp_dropout=args.mlp_dropout,
        temp=args.temp,
        sinkhorn_iters=args.sinkhorn_iters,
        top_k_sparse=None,
        weight_threshold=None,
        use_distance=args.use_distance,
        sparse_chunk_t=args.sparse_chunk_t,
        dist_penalty_lambda=args.dist_penalty_lambda,
    )

    model = GeoGNNOT(
        in_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        sinkhorn_cfg=sinkhorn_cfg,
        feat_embed_cfg=feat_embed_cfg,
        use_distance=args.use_distance,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    out_dir = Path(args.out_dir)
    ckpt_out = Path(args.ckpt_out)

    # Early stopping init
    best_mse = float("inf")
    patience_counter = 0
    history = []
    cand_src_idx_cached = cand_src_idx
    cand_dist_cached = cand_dist
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        if args.cand_method == "score_topk" and (epoch == 1 or (args.cand_refresh_every > 0 and (epoch - 1) % args.cand_refresh_every == 0)):
            model.eval()
            with torch.no_grad():
                h = model.encoder(x, edge_index)
                hs = {k: h[k][source_idx] for k in h.keys()}
                ht = {k: h[k][target_idx] for k in h.keys()}

                Ns = int(source_idx.numel())
                Nt = int(target_idx.numel())
                K = min(args.cand_k_src, Ns)
                src_chunk = max(1, int(args.score_topk_src_chunk))

                best_scores = torch.full((Nt, K), float("-inf"), device=device)
                best_idx = torch.zeros((Nt, K), dtype=torch.long, device=device)
                ht_by_mod = {mn: ht[mn] for mn in ht.keys()}

                for s0 in range(0, Ns, src_chunk):
                    s1 = min(Ns, s0 + src_chunk)
                    bs = s1 - s0
                    feat_parts = []
                    for mn in sorted(hs.keys()):
                        hs_blk = hs[mn][s0:s1]
                        ht_m = ht_by_mod[mn]
                        hs_rep = hs_blk[None, :, :].expand(Nt, bs, hs_blk.shape[1])
                        ht_rep = ht_m[:, None, :].expand(Nt, bs, ht_m.shape[1])
                        feat_parts.append(
                            torch.cat([hs_rep.reshape(Nt * bs, -1), ht_rep.reshape(Nt * bs, -1)], dim=-1)
                        )
                    feat = torch.cat(feat_parts, dim=1)
                    if args.use_distance:
                        expected_in = model.predictor.mlp.layers[0].in_features
                        missing = int(expected_in - feat.shape[1])
                        if missing < 0:
                            raise ValueError(
                                f"score_topk feat dim ({feat.shape[1]}) exceeds predictor MLP input dim ({expected_in})."
                            )
                        if missing > 0:
                            feat = torch.cat(
                                [feat, torch.zeros((feat.shape[0], missing), device=feat.device, dtype=feat.dtype)],
                                dim=1,
                            )
                    scores_blk = model.predictor.mlp(feat).reshape(Nt, bs)

                    merged_scores = torch.cat([best_scores, scores_blk], dim=1)  # (Nt, K+bs)
                    merged_idx = torch.cat(
                        [best_idx, torch.arange(s0, s1, device=device)[None, :].expand(Nt, bs)],
                        dim=1,
                    )
                    new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1)
                    best_scores = new_scores
                    best_idx = torch.gather(merged_idx, 1, new_pos)

                cand_src_idx_cached = best_idx  # (Nt,K) in source-position space
                if args.use_distance:
                    cand_dist_np = geo_csr_gather_candidate_distances(
                        meta["geo_csr"], meta["source_idx"], meta["target_valid_idx"], cand_src_idx_cached.cpu().numpy()
                    )
                    cand_dist_cached = torch.from_numpy(cand_dist_np).to(device)
                    cand_dist_cached = torch.nan_to_num(cand_dist_cached, nan=0.0, posinf=1e6, neginf=0.0)
                else:
                    cand_dist_cached = None
            model.train()

        out = model(
            x=x,
            edge_index=edge_index,
            tuning=tuning_normalized,
            source_idx=source_idx,
            target_idx=target_idx,
            cand_src_idx=cand_src_idx_cached,
            cand_dist=cand_dist_cached,
        )

        tuning_pred = out["tuning_pred"]
        tuning_target = tuning_normalized[target_idx]
        loss_mse = nn.functional.mse_loss(tuning_pred, tuning_target)

        loss_topo = torch.tensor(0.0, device=device)
        if topo_loss_fn is not None:
             W_soft = out["W"]
             real_src_coords = coords[source_idx]
             loss_topo = topo_loss_fn(W_soft, cand_src_idx_cached, real_src_coords)
             
        loss = loss_mse + args.topo_lambda * loss_topo
        loss.backward()

        if args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

        optimizer.step()
        scheduler.step()

        mse = float(loss.item())
        history.append({"epoch": epoch, "train_mse_zscore": mse, "lr": scheduler.get_last_lr()[0]})
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs}: train_mse_zscore={mse:.6f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping check
        if args.patience > 0:
            if mse < best_mse:
                best_mse = mse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[EarlyStopping] Stop at epoch {epoch}: no improvement for {args.patience} epochs (best_mse_zscore={best_mse:.6f}).")
                    break

    # If grid search (no ckpt save), save hparams to json if MSE is good.
    if args.grid_candidates_out:
        if args.max_train_mse is not None and mse > args.max_train_mse:
            print(f"[grid_skip] MSE {mse:.6f} > max_train_mse {args.max_train_mse:.6f}, not saving candidate.")
            return

        out_path = Path(args.grid_candidates_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hparams = {
            "temp": args.temp,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "clip_norm": args.clip_norm,
            "cand_k_src": args.cand_k_src,
            "topo_lambda": args.topo_lambda,
            "train_mse": mse,
        }
        
        existing = []
        if out_path.exists():
            try:
                with open(out_path, "r") as f:
                    existing = json.load(f)
            except:
                existing = []
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(hparams)
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"[ok] saved grid candidate to {out_path}")

    ckpt_payload = {
        "state_dict": model.state_dict(),
        "config": {
            "cfg": {
                "hidden_dim": args.hidden_dim,
                "gat_heads": args.gat_heads,
                "gat_dropout": args.gat_dropout,
                "sinkhorn": {
                    "mlp_hidden": args.mlp_hidden,
                    "mlp_dropout": args.mlp_dropout,
                    "temp": args.temp,
                    "sinkhorn_iters": args.sinkhorn_iters,
                    "sparse_chunk_t": args.sparse_chunk_t,
                    "dist_penalty_lambda": args.dist_penalty_lambda,
                },
                "feat_embed_cfg": {
                    "use_intrinsic": args.use_intrinsic,
                    "use_normal": args.use_normal,
                    "use_xyz": args.use_xyz,
                    "intrinsic_dim": 32,
                    "normal_dim": 32,
                    "xyz_dim": 32,
                },
                "use_distance": args.use_distance,
            },
            "feat_norm": asdict(feat_norm) if feat_norm else None,
            "tuning_stats": tuning_stats,
            "module_name": module_name,
            "template_subject": spec.subject,
        },
    }

    ckpt_out_s = str(ckpt_out).replace("\\", "/")
    save_artifacts = "/geognn_ot/results/checkpoints/" in ckpt_out_s or "/results/checkpoints/" in ckpt_out_s

    if save_artifacts:
        ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(ckpt_payload, ckpt_out)
        print(f"Saved checkpoint: {ckpt_out}")

        hist_path = ckpt_out.with_suffix(".history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved history: {hist_path}")
    else:
        print(f"Skipped saving artifacts (checkpoint/history) for non-checkpoints run: {ckpt_out}")


def cmd_evaluate(args):
    repo_root = Path(args.repo_root).resolve()
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]["cfg"]
    module_name = ckpt["config"].get("module_name", "unknown")
    ckpt_stem = ckpt_path.stem
    ckpt_tag = ckpt_stem
    for h in ("lh", "rh"):
        prefix = f"R1_{h}_"
        if ckpt_stem.startswith(prefix):
            ckpt_tag = ckpt_stem[len(prefix) :]
            break

    
    # Output root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_embed_cfg = NodeFeatureEmbedConfig(
        use_intrinsic=args.use_intrinsic or cfg["feat_embed_cfg"]["use_intrinsic"],
        use_normal=args.use_normal or cfg["feat_embed_cfg"]["use_normal"],
        use_xyz=args.use_xyz or cfg["feat_embed_cfg"]["use_xyz"],
        intrinsic_dim=cfg["feat_embed_cfg"]["intrinsic_dim"],
        normal_dim=cfg["feat_embed_cfg"]["normal_dim"],
        xyz_dim=cfg["feat_embed_cfg"]["xyz_dim"],
    )

    sinkhorn_kwargs = {k: v for k, v in cfg["sinkhorn"].items() if k in SinkhornPredictorConfig.__annotations__}
    sinkhorn_cfg = SinkhornPredictorConfig(**sinkhorn_kwargs)
    sinkhorn_cfg.use_distance = args.use_distance or cfg["use_distance"]
    if args.dist_penalty_lambda is not None:
        sinkhorn_cfg.dist_penalty_lambda = args.dist_penalty_lambda
    sinkhorn_cfg.sparse_chunk_t = args.sparse_chunk_t

    in_dim = 0
    if feat_embed_cfg.use_intrinsic:
        in_dim += 2
    if feat_embed_cfg.use_normal:
        in_dim += 3
    if feat_embed_cfg.use_xyz:
        in_dim += 3

    model = GeoGNNOT(
        in_dim=in_dim,
        hidden_dim=cfg["hidden_dim"],
        gat_heads=cfg["gat_heads"],
        gat_dropout=cfg["gat_dropout"],
        sinkhorn_cfg=sinkhorn_cfg,
        feat_embed_cfg=feat_embed_cfg,
        use_distance=cfg["use_distance"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tuning_stats = ckpt["config"].get("tuning_stats", ckpt["config"].get("tuning_minmax", None))
    tuning_norm_type = tuning_stats.get("norm_type", "minmax") if tuning_stats and isinstance(tuning_stats, dict) else "minmax"
    tmin = np.array(tuning_stats["min"], dtype=np.float32) if tuning_stats and "min" in tuning_stats else None
    tmax = np.array(tuning_stats["max"], dtype=np.float32) if tuning_stats and "max" in tuning_stats else None
    tmu = np.array(tuning_stats["mu"], dtype=np.float32) if tuning_stats and "mu" in tuning_stats else None
    tstd = np.array(tuning_stats["std"], dtype=np.float32) if tuning_stats and "std" in tuning_stats else None
    teps = float(tuning_stats.get("eps", 1e-8)) if tuning_stats and isinstance(tuning_stats, dict) else 1e-8

    # Robustly prepare stats as tensors once
    tmin_t = torch.from_numpy(tmin).to(device).reshape(-1) if tmin is not None else None
    tmax_t = torch.from_numpy(tmax).to(device).reshape(-1) if tmax is not None else None
    tmu_t = torch.from_numpy(tmu).to(device).reshape(-1) if tmu is not None else None
    tstd_t = torch.from_numpy(tstd).to(device).reshape(-1) if tstd is not None else None

    feat_norm_cfg = ckpt["config"].get("feat_norm", None)
    feat_norm = None
    if feat_norm_cfg is not None:
        feat_norm = FeatureNormStats(
            intrinsic_mu=np.array(feat_norm_cfg["intrinsic_mu"], dtype=np.float32),
            intrinsic_std=np.array(feat_norm_cfg["intrinsic_std"], dtype=np.float32),
            xyz_mu=np.array(feat_norm_cfg["xyz_mu"], dtype=np.float32),
            xyz_std=np.array(feat_norm_cfg["xyz_std"], dtype=np.float32),
            eps=float(feat_norm_cfg.get("eps", 1e-8)),
        )

    template_subject = args.template_subject or ckpt["config"].get("template_subject", "R1")
    results = {}

    for subject in args.subjects:
        for hemi in args.hemis:
            try:
                need_geo = (
                    (args.use_distance or cfg["use_distance"])
                    or sinkhorn_cfg.dist_penalty_lambda > 0.0
                )
                spec = SubjectSpec(subject=subject, hemi=hemi)
                data_loader = PreprocessedSubject(spec=spec, root=repo_root, use_distances=need_geo)
                data, meta = data_loader.load()

                # Defensive: ensure indices are in-bounds even if preprocessing artifacts are inconsistent.
                n_eff = min(int(data.x.shape[0]), int(data.tuning.shape[0]), int(data.mid_coords.shape[0]))
                src_np = meta["source_idx"].astype(np.int64, copy=False)
                tgt_np = meta["target_valid_idx"].astype(np.int64, copy=False)
                src_np = src_np[(src_np >= 0) & (src_np < n_eff)]
                tgt_np = tgt_np[(tgt_np >= 0) & (tgt_np < n_eff)]
                source_idx = torch.from_numpy(src_np).to(device)
                target_idx = torch.from_numpy(tgt_np).to(device)
                edge_index = data.edge_index.to(device)
                tuning = data.tuning.to(device)
                
                src_tuning_S = tuning[source_idx]
                smu_t = src_tuning_S.mean(dim=0)
                sstd_t = src_tuning_S.std(dim=0).clamp_min(teps)
                smin_t = src_tuning_S.min(dim=0).values
                smax_t = src_tuning_S.max(dim=0).values

                x_np = data.x.cpu().numpy()
                coords_np = data.mid_coords.cpu().numpy()
                area_np = data.area.cpu().numpy()

                coords_mds = _align_mds_like_baseline(np.asarray(x_np[:, 0:2], dtype=float), area_np)

                pkl_dict = None
                try:
                    pkl_dict = _load_subject_pkl(repo_root, subject, hemi)
                except Exception as e:
                    print(f"[warn] could not load PKL for plotting ({subject}_{hemi}): {e}")

                tmpl_data = None
                if subject != template_subject:
                    tmpl_spec = SubjectSpec(subject=template_subject, hemi=hemi)
                    tmpl_loader = PreprocessedSubject(spec=tmpl_spec, root=repo_root, use_distances=False)
                    tmpl_data, _ = tmpl_loader.load()
                    coords_np = align_with_node_correspondence(
                        template_mid_coords=tmpl_data.mid_coords.cpu().numpy(),
                        template_global_vertex_idx=tmpl_data.global_vertex_idx.cpu().numpy(),
                        subject_mid_coords=coords_np,
                        subject_global_vertex_idx=data.global_vertex_idx.cpu().numpy(),
                        template_subject=template_subject,
                        template_hemi=hemi,
                        subject_subject=subject,
                        subject_hemi=hemi,
                        repo_root=repo_root,
                    )

                    normals = compute_vertex_normals_numpy(coords_np, data.faces_sub.cpu().numpy(), eps=1e-8)
                    x_np[:, 2:5] = normals

                coords_3d_vis = np.asarray(coords_np, dtype=float)

                if feat_norm is not None:
                    x_np, coords_np = apply_feature_norm_numpy(x_np, coords_np, feat_norm, unit_normal=True)

                x_norm = torch.from_numpy(x_np).to(device)
                coords = torch.from_numpy(coords_np).to(device)

                x_parts = []
                if feat_embed_cfg.use_intrinsic:
                    x_parts.append(x_norm[:, 0:2])
                if feat_embed_cfg.use_normal:
                    x_parts.append(x_norm[:, 2:5])
                if feat_embed_cfg.use_xyz:
                    x_parts.append(coords)
                x = torch.cat(x_parts, dim=-1)

                if tuning_stats is not None:
                    if tuning_norm_type == "zscore" and tmu_t is not None and tstd_t is not None:
                        tuning_normalized = (tuning - tmu_t[None, :]) / tstd_t.clamp_min(teps)[None, :]
                    elif tmin_t is not None and tmax_t is not None:
                        denom = (tmax_t - tmin_t).clamp_min(teps)
                        tuning_normalized = (tuning - tmin_t[None, :]) / denom[None, :]
                    else:
                        tuning_normalized = tuning
                else:
                    tuning_normalized = tuning

                cand_src_idx = None
                cand_dist = None
                if (args.use_distance or cfg["use_distance"]) and meta["geo_csr"] is None:
                    raise ValueError("use_distance=True requires geodesic distances (geo_csr).")

                if args.cand_method == "score_topk":
                    with torch.no_grad():
                        h = model.encoder(x, edge_index)
                        hs = {k: h[k][source_idx] for k in h.keys()}
                        ht = {k: h[k][target_idx] for k in h.keys()}

                        Ns = int(source_idx.numel())
                        Nt = int(target_idx.numel())
                        K = min(args.cand_k_src, Ns)
                        src_chunk = max(1, int(args.score_topk_src_chunk))

                        best_scores = torch.full((Nt, K), float("-inf"), device=device)
                        best_idx = torch.zeros((Nt, K), dtype=torch.long, device=device)
                        ht_by_mod = {mn: ht[mn] for mn in ht.keys()}

                        for s0 in range(0, Ns, src_chunk):
                            s1 = min(Ns, s0 + src_chunk)
                            bs = s1 - s0
                            feat_parts = []
                            for mn in sorted(hs.keys()):
                                hs_blk = hs[mn][s0:s1]
                                ht_m = ht_by_mod[mn]
                                hs_rep = hs_blk[None, :, :].expand(Nt, bs, hs_blk.shape[1])
                                ht_rep = ht_m[:, None, :].expand(Nt, bs, ht_m.shape[1])
                                feat_parts.append(
                                    torch.cat([hs_rep.reshape(Nt * bs, -1), ht_rep.reshape(Nt * bs, -1)], dim=-1)
                                )
                            feat = torch.cat(feat_parts, dim=1)
                            if cfg["use_distance"]:
                                expected_in = model.predictor.mlp.layers[0].in_features
                                missing = int(expected_in - feat.shape[1])
                                if missing < 0:
                                    raise ValueError(
                                        f"score_topk feat dim ({feat.shape[1]}) exceeds predictor MLP input dim ({expected_in})."
                                    )
                                if missing > 0:
                                    feat = torch.cat(
                                        [feat, torch.zeros((feat.shape[0], missing), device=feat.device, dtype=feat.dtype)],
                                        dim=1,
                                    )
                            scores_blk = model.predictor.mlp(feat).reshape(Nt, bs)

                            merged_scores = torch.cat([best_scores, scores_blk], dim=1)
                            merged_idx = torch.cat(
                                [best_idx, torch.arange(s0, s1, device=device)[None, :].expand(Nt, bs)],
                                dim=1,
                            )
                            new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1)
                            best_scores = new_scores
                            best_idx = torch.gather(merged_idx, 1, new_pos)

                        cand_src_idx = best_idx
                        if cfg["use_distance"]:
                            cand_dist_np = geo_csr_gather_candidate_distances(
                                meta["geo_csr"], meta["source_idx"], meta["target_valid_idx"], cand_src_idx.cpu().numpy()
                            )
                            cand_dist = torch.from_numpy(cand_dist_np).to(device)
                            cand_dist = torch.nan_to_num(cand_dist, nan=0.0, posinf=1e6, neginf=0.0)
                else:
                    raise ValueError(f"Unknown cand_method: {args.cand_method}")

                with torch.no_grad():
                    out = model(
                        x=x,
                        edge_index=edge_index,
                        tuning=tuning_normalized,
                        source_idx=source_idx,
                        target_idx=target_idx,
                        cand_src_idx=cand_src_idx,
                        cand_dist=cand_dist,
                    )

                tuning_pred = out["tuning_pred"]
                tuning_target_norm = tuning_normalized[target_idx]
                mse_norm = float(nn.functional.mse_loss(tuning_pred, tuning_target_norm).item())

                tuning_pred_raw = tuning_pred * tstd_t[None, :] + tmu_t[None, :]
                tuning_target_raw = tuning_target_norm * tstd_t[None, :] + tmu_t[None, :]

                pred_z_S = (tuning_pred_raw - smu_t[None, :]) / sstd_t[None, :]
                true_z_S = (tuning_target_raw - smu_t[None, :]) / sstd_t[None, :]
                mse_zscore = float(nn.functional.mse_loss(pred_z_S, true_z_S).item())

                sdenom = (smax_t - smin_t).clamp_min(teps)
                pred_mm_S = (tuning_pred_raw - smin_t[None, :]) / sdenom[None, :]
                true_mm_S = (tuning_target_raw - smin_t[None, :]) / sdenom[None, :]
                mse_minmax = float(nn.functional.mse_loss(pred_mm_S, true_mm_S).item())

                print(f"[{subject} {hemi}] MSE (Z-score): {mse_zscore:.6f}, MSE (Min-Max): {mse_minmax:.6f}")
                mse = mse_zscore

                eval_out_dir = Path(args.out_dir) / subject / hemi
                eval_out_dir.mkdir(parents=True, exist_ok=True)

                base = f"{subject}_{hemi}"

                if tuning_stats is not None:
                    if tuning_norm_type == "zscore" and tmu_t is not None and tstd_t is not None:
                        tuning_true_den = tuning_target_norm * tstd_t[None, :] + tmu_t[None, :]
                        tuning_pred_den = tuning_pred * tstd_t[None, :] + tmu_t[None, :]
                    elif tmin_t is not None and tmax_t is not None:
                        denom = (tmax_t - tmin_t).clamp_min(teps)
                        tuning_true_den = tuning_target_norm * denom[None, :] + tmin_t[None, :]
                        tuning_pred_den = tuning_pred * denom[None, :] + tmin_t[None, :]
                    else:
                        tuning_true_den = tuning_target_norm
                        tuning_pred_den = tuning_pred
                else:
                    tuning_true_den = tuning_target_norm
                    tuning_pred_den = tuning_pred

                W_np = out["W"].detach().cpu().numpy()
                np.savez_compressed(eval_out_dir / f"W_{base}.npz", W=W_np)

                node_ids = data.global_vertex_idx.detach().cpu().numpy().astype(np.int64, copy=False)
                tgt_ids = node_ids[tgt_np]
                pred_np = tuning_pred_den.detach().cpu().numpy()
                true_np = tuning_true_den.detach().cpu().numpy()
                tsv_path = eval_out_dir / f"predicted_{base}.tsv"
                with open(tsv_path, "w") as f:
                    f.write("Node_ID\tPred_0\tPred_1\tTrue_0\tTrue_1\n")
                    for nid, p0, p1, t0, t1 in zip(
                        tgt_ids.astype(np.int64, copy=False),
                        pred_np[:, 0].astype(np.float32, copy=False),
                        pred_np[:, 1].astype(np.float32, copy=False),
                        true_np[:, 0].astype(np.float32, copy=False),
                        true_np[:, 1].astype(np.float32, copy=False),
                    ):
                        f.write(f"{int(nid)}\t{float(p0)}\t{float(p1)}\t{float(t0)}\t{float(t1)}\n")

                N_all = int(coords_np.shape[0])
                if tmin_t is not None and tmax_t is not None:
                    denom = (tmax_t - tmin_t).clamp_min(teps)
                    tuning_true_all = (tuning_normalized * denom[None, :] + tmin_t[None, :]).detach().cpu().numpy()
                else:
                    tuning_true_all = tuning_normalized.detach().cpu().numpy()
                if tuning_true_all.shape[0] != N_all:
                    raise ValueError(f"tuning_true_all length {tuning_true_all.shape[0]} != coords length {N_all}")

                tuning_pred_all = np.asarray(tuning_true_all, dtype=np.float32).copy()
                tuning_pred_all[tgt_np] = pred_np
                v1_mask = np.zeros((N_all,), dtype=bool)
                v1_mask[src_np] = True

                plot_mask = data.has_tuning.detach().cpu().numpy().astype(bool)
                if plot_mask.shape[0] != N_all:
                    plot_mask = np.ones((N_all,), dtype=bool)

                if pkl_dict is not None:
                    node_ids_all = data.global_vertex_idx.detach().cpu().numpy()
                    coords_loc_all = _coords_from_pkl_loc(pkl_dict, node_ids_all)
                    tuning_pkl_all = _tuning_from_pkl(pkl_dict, node_ids_all)
                    is_center_all = _is_center_from_pkl(pkl_dict, node_ids_all)

                    coords_plot_raw = coords_loc_all[plot_mask]
                    area_plot = area_np[plot_mask]
                    center_idx = None
                    center_mask_plot = is_center_all[plot_mask]
                    if np.any(center_mask_plot):
                        center_idx = int(np.where(center_mask_plot)[0][0])
                    coords_plot = _align_mds_like_baseline(coords_plot_raw, area_plot, center_idx=center_idx)

                    tuning_true_plot = tuning_pkl_all[plot_mask]
                    tuning_pred_plot_all = tuning_pkl_all.copy()
                    tuning_pred_plot_all[tgt_np] = pred_np
                    tuning_pred_plot = tuning_pred_plot_all[plot_mask]
                else:
                    coords_plot = coords_mds[plot_mask]
                    tuning_true_plot = tuning_true_all[plot_mask]
                    tuning_pred_plot = tuning_pred_all[plot_mask]

                v1_mask_plot = v1_mask[plot_mask]
                area_plot = area_np[plot_mask]

                from TUNING_COLOR_UTILS import compute_tuning_colors
                colors_for_html = compute_tuning_colors(tuning_true_plot, v1_mask=v1_mask_plot, tag=hemi)

                _save_tuning_compare_png(
                    out_png=eval_out_dir / f"{subject}_{hemi}_tuning_compare.png",
                    coords_xy=coords_plot,
                    tuning_true_all=tuning_true_plot,
                    tuning_pred_all=tuning_pred_plot,
                    v1_mask=v1_mask_plot,
                    tag=hemi,
                    data_name=subject,
                )

                # 4) Plotly HTML of 3D coords colored by area (rotatable)
                _save_align_html_png(
                    out_html=eval_out_dir / f"align_{subject}_{hemi}.html",
                    out_png=eval_out_dir / f"align_{subject}_{hemi}.png",
                    coords_3d=coords_3d_vis[plot_mask],
                    area_labels=area_plot,
                    colors=colors_for_html,
                    title=f"{subject}_{hemi} 3D Alignment (colored by area)",
                    template_coords_3d=(
                        tmpl_data.mid_coords.detach().cpu().numpy()[tmpl_data.has_tuning.detach().cpu().numpy().astype(bool)]
                        if tmpl_data is not None
                        else None
                    ),
                    template_area_labels=(
                        tmpl_data.area.detach().cpu().numpy()[tmpl_data.has_tuning.detach().cpu().numpy().astype(bool)]
                        if tmpl_data is not None
                        else None
                    ),
                    template_name=f"Brain A ({template_subject})",
                    subject_name=f"Brain B ({subject})",
                )

                result_file = eval_out_dir / "results.json"
                with open(result_file, "w") as f:
                    json.dump(
                        {
                            "mse_minmax": mse,
                            "subject": subject,
                            "hemi": hemi,
                            "module": module_name,
                            "ckpt": str(ckpt_path),
                            "files": {
                                "W": f"W_{base}.npz",
                                "predicted_tsv": f"predicted_{base}.tsv",
                                "tuning_png": f"{subject}_{hemi}_tuning_compare.png",
                                "align_html": f"align_{subject}_{hemi}.html",
                                "align_png": f"align_{subject}_{hemi}.png",
                            },
                        },
                        f,
                        indent=2,
                    )

                results[f"{subject}_{hemi}"] = mse
                print(f"{subject}_{hemi}: mse_minmax={mse:.6f}")
            except Exception as e:
                import traceback

                print(f"Error evaluating {subject}_{hemi}: {e}")
                print(traceback.format_exc())
                results[f"{subject}_{hemi}"] = None

    summary_file = Path(args.out_dir) / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    args = parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()

