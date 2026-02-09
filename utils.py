import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import torch

def _rotate_to_align_x(xs: np.ndarray, ys: np.ndarray, areas: np.ndarray):
    xs_np = np.asarray(xs, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    areas_np = np.asarray(areas, dtype=int)
    unique_areas = np.unique(areas_np)
    centroids = []
    for a in unique_areas:
        mask = areas_np == a
        if not np.any(mask):
            continue
        centroids.append([xs_np[mask].mean(), ys_np[mask].mean()])
    if len(centroids) < 2:
        return xs_np, ys_np, 0.0, np.array([xs_np.mean() if xs_np.size else 0.0, ys_np.mean() if ys_np.size else 0.0])
    C = np.array(centroids, dtype=float)
    C_centered = C - C.mean(axis=0)
    cov = np.cov(C_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle = np.arctan2(principal[1], principal[0])
    cos_t, sin_t = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    P = np.stack([xs_np, ys_np], axis=1) @ R.T
    center_rot = (C.mean(axis=0)) @ R.T
    return P[:, 0], P[:, 1], angle, center_rot


def _rotate_by_angle(xs: np.ndarray, ys: np.ndarray, delta_rad: float):
    xs_np = np.asarray(xs, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    R = np.array([[c, -s], [s, c]])
    P = np.stack([xs_np, ys_np], axis=1) @ R.T
    return P[:, 0], P[:, 1]


def build_df(
    subject: str,
    hemi: str,
    global_vertex_idx: np.ndarray,
    area: np.ndarray,
    tuning: np.ndarray,
    mid_coords: np.ndarray,
) -> pd.DataFrame:
    # For visualization we want a stable 2D embedding.
    # Here we use mid-surface coordinates projected to (x,y). Any final translation
    # (e.g., is_center alignment) is handled in the plotting function to match baseline_new.
    xy = np.asarray(mid_coords, dtype=np.float32)[:, :2]
    return pd.DataFrame(
        {
            "nodeIdx": global_vertex_idx.astype(int),
            "area": area.astype(int),
            "x": xy[:, 0].astype(float),
            "y": xy[:, 1].astype(float),
            "tuningX": tuning[:, 0].astype(float),
            "tuningY": tuning[:, 1].astype(float),
            "subject": subject,
            "hemi": hemi,
        }
    )


def save_outputs(
    out_dir: Path,
    *,
    subject: str,
    hemi: str,
    df: pd.DataFrame,
    W_st: np.ndarray,
    pred_t_t: np.ndarray,
    true_t_t: np.ndarray,
    target_nodeIdx: np.ndarray,
    source_nodeIdx: np.ndarray,
    has_tuning_dict: Optional[dict[int, bool]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    algo = "geognn_ot"
    base = f"{subject}_{hemi}_{algo}"

    # TSV: baseline_new-style (V2–V4 only). Do NOT write V1 rows.
    v1_set = set(df.loc[df["area"] == 1, "nodeIdx"].astype(int).tolist())

    tsv_rows = []
    for i in range(len(target_nodeIdx)):
        if int(target_nodeIdx[i]) in v1_set:
            continue
        tsv_rows.append(
            {
                "Node_ID": int(target_nodeIdx[i]),
                "Pred_0": float(pred_t_t[i, 0]),
                "Pred_1": float(pred_t_t[i, 1]),
                "True_0": float(true_t_t[i, 0]),
                "True_1": float(true_t_t[i, 1]),
            }
        )
    pd.DataFrame(tsv_rows).to_csv(out_dir / f"predicted_{base}.tsv", sep="\t", index=False)

    np.savez_compressed(
        out_dir / f"W_{base}.npz",
        W=W_st.astype(np.float32),
        source_nodeIdx=source_nodeIdx.astype(np.int64),
        target_nodeIdx=target_nodeIdx.astype(np.int64),
    )

    np.savez_compressed(
        out_dir / "record.npz",
        W=W_st.astype(np.float32),
        pred=pred_t_t.astype(np.float32),
        true=true_t_t.astype(np.float32),
        source_nodeIdx=source_nodeIdx.astype(np.int64),
        target_nodeIdx=target_nodeIdx.astype(np.int64),
    )

    # PNG: baseline_new-style 2-panel plot
    try:
        import matplotlib.pyplot as plt
        from TUNING_COLOR_UTILS import compute_tuning_colors, get_tuning_colormap, should_flip_y_red_bottom

        node_ids_full = df["nodeIdx"].to_numpy(dtype=np.int64)
        areas_full = df["area"].to_numpy(dtype=np.int64)
        xs_full = df["x"].to_numpy(dtype=float)
        ys_full = df["y"].to_numpy(dtype=float)
        tx_full = df["tuningX"].to_numpy(dtype=float)
        ty_full = df["tuningY"].to_numpy(dtype=float)

        finite_xy = np.isfinite(xs_full) & np.isfinite(ys_full)
        finite_t = np.isfinite(tx_full) & np.isfinite(ty_full)
        area_mask = np.isin(areas_full, np.array([1, 2, 3, 4], dtype=np.int64))
        plot_mask = finite_xy & finite_t & area_mask
        if not np.any(plot_mask):
            raise ValueError("No nodes with finite loc+tuning (V1–V4) available for plotting.")

        node_ids = node_ids_full[plot_mask]
        areas = areas_full[plot_mask]
        xs = xs_full[plot_mask]
        ys = ys_full[plot_mask]
        true_tuning = np.stack([tx_full[plot_mask], ty_full[plot_mask]], axis=1).astype(float)

        # 1) PCA-based alignment (on plotted nodes)
        xs_rot, ys_rot, _, _ = _rotate_to_align_x(xs, ys, areas)
        v1_mask = (areas == 1)
        if np.any(v1_mask):
            cx = float(xs_rot[v1_mask].mean())
            cy = float(ys_rot[v1_mask].mean())
            cur_angle = np.arctan2(cy, cx)
            delta = np.pi - cur_angle
            xs_rot, ys_rot = _rotate_by_angle(xs_rot, ys_rot, delta)

        is_center_idx = None
        if "is_center" in df.columns:
            center_mask_full = df["is_center"].to_numpy(dtype=int) == 1
            if np.any(center_mask_full & plot_mask):
                # map full index -> plotted index
                full_idx = int(np.where(center_mask_full & plot_mask)[0][0])
                is_center_idx = int(np.sum(plot_mask[:full_idx]))
        coords = np.stack([xs_rot, ys_rot], axis=1).astype(float)
        if is_center_idx is not None:
            coords = coords - coords[is_center_idx : is_center_idx + 1, :]
        else:
            coords = coords - coords.mean(axis=0, keepdims=True)

        # Build predicted tuning aligned to plotted node_ids
        pred_map: dict[int, np.ndarray] = {}
        # V1: always true tuning (both panels)
        for nid, a, txy in zip(node_ids, areas, true_tuning):
            if int(a) == 1:
                pred_map[int(nid)] = np.array([float(txy[0]), float(txy[1])], dtype=float)
        # V2–V4: fill from model predictions
        for nid, (px, py) in zip(target_nodeIdx.astype(np.int64), pred_t_t):
            pred_map[int(nid)] = np.array([float(px), float(py)], dtype=float)

        pred_tuning = np.array([pred_map.get(int(nid), true_tuning[i]) for i, nid in enumerate(node_ids)], dtype=float)

        # 3) colors (baseline_new logic), computed on the plotted set only
        true_colors = compute_tuning_colors(true_tuning, v1_mask=v1_mask, tag=hemi)
        pred_colors = compute_tuning_colors(pred_tuning, v1_mask=v1_mask, tag=hemi)
        pred_colors = np.asarray(pred_colors, dtype=float).copy()
        pred_colors[v1_mask] = np.asarray(true_colors, dtype=float)[v1_mask]

        # 5. Final orientation
        if np.any(v1_mask):
            try:
                true_c_for_flip = np.round(np.asarray(true_colors, dtype=float) * 10) / 10.0
                true_c_for_flip = np.clip(true_c_for_flip, 0.0, 1.0)
                flip_y = should_flip_y_red_bottom(coords[v1_mask], true_c_for_flip[v1_mask])
                if flip_y:
                    coords = coords.copy()
                    coords[:, 1] *= -1.0
            except Exception:
                pass

        # 6. Map to RGBA
        true_c = np.round(np.asarray(true_colors, dtype=float) * 10) / 10.0
        pred_c = np.round(np.asarray(pred_colors, dtype=float) * 10) / 10.0
        true_c = np.clip(true_c, 0.0, 1.0)
        pred_c = np.clip(pred_c, 0.0, 1.0)
        
        cmap = get_tuning_colormap()
        true_rgba = [cmap(c) for c in true_c]
        pred_rgba = [cmap(c) for c in pred_c]

        unconnected_vn_set: set[int] = set()
        try:
            col_sums = np.sum(np.asarray(W_st, dtype=float), axis=0)
            for j, nid in enumerate(target_nodeIdx.astype(np.int64)):
                if j < col_sums.shape[0] and float(col_sums[j]) == 0.0:
                    idx_arr = np.where(node_ids == int(nid))[0]
                    if idx_arr.size:
                        unconnected_vn_set.add(int(idx_arr[0]))
        except Exception:
            pass

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        def plot_with_mask(ax, rgba, title: str, *, is_predicted_panel: bool) -> None:
            for a in np.unique(areas):
                idxs = np.where(areas == a)[0]
                if idxs.size == 0:
                    continue
                if int(a) == 1:
                    # V1: circle
                    ax.scatter(
                        coords[idxs, 0],
                        coords[idxs, 1],
                        c=[rgba[i] for i in idxs],
                        s=6,
                        alpha=1.0,
                        linewidth=0,
                        marker="o",
                    )
                else:
                    connected_idxs = [i for i in idxs if i not in unconnected_vn_set]
                    unconnected_idxs = [i for i in idxs if i in unconnected_vn_set]
                    if connected_idxs:
                        # V2–V4: diamond
                        ax.scatter(
                            coords[connected_idxs, 0],
                            coords[connected_idxs, 1],
                            c=[rgba[i] for i in connected_idxs],
                            s=6,
                            alpha=1.0,
                            linewidth=0,
                            marker="D",
                        )
                    if unconnected_idxs:
                        color_unconn = "black" if is_predicted_panel else [rgba[i] for i in unconnected_idxs]
                        ax.scatter(
                            coords[unconnected_idxs, 0],
                            coords[unconnected_idxs, 1],
                            c=color_unconn,
                            s=6,
                            alpha=1.0,
                            linewidth=0,
                            marker="D",
                        )
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.axis("off")

        plot_with_mask(axes[0], true_rgba, "True", is_predicted_panel=False)
        plot_with_mask(axes[1], pred_rgba, "Predicted", is_predicted_panel=True)
        plt.tight_layout()
        fig.savefig(out_dir / f"{subject}_{hemi}_tuning_compare.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[warn] png save failed: {e}")
