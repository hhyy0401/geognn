"""
Alignment functions for matching coordinates between different subjects.

This module provides node correspondence-based alignment that uses PKL node IDs
to find overlapping nodes and apply Thin-Plate Spline (TPS) transformation
using only the overlapping nodes.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import RBFInterpolator
    HAS_RBF = True
except ImportError:
    HAS_RBF = False


def load_pkl_node_mapping(
    subject: str, hemi: str, repo_root: Path
) -> Tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Load node ID to location mapping from PKL file.
    
    Returns:
        node_to_loc: dict mapping node_id -> loc (2D MDS coordinates)
        node_to_loc_3d: dict mapping node_id -> loc_3D (3D coordinates, if available)
    """
    pkl_paths = [
        repo_root / "data" / f"{subject}_{hemi}.pkl",
        repo_root.parent / "data" / f"{subject}_{hemi}.pkl",
    ]
    
    for pkl_path in pkl_paths:
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            
            node_to_loc = {}
            node_to_loc_3d = {}
            
            for node_id, node_data in pkl_data.items():
                if not isinstance(node_data, dict):
                    continue
                
                node_id_int = int(node_id)
                
                # Load 2D MDS location
                if "loc" in node_data:
                    loc = node_data["loc"]
                    if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                        node_to_loc[node_id_int] = np.array([float(loc[0]), float(loc[1])], dtype=np.float32)
                
                # Load 3D location if available
                if "loc_3D" in node_data:
                    loc_3d = node_data["loc_3D"]
                    if isinstance(loc_3d, (tuple, list)) and len(loc_3d) >= 3:
                        node_to_loc_3d[node_id_int] = np.array(
                            [float(loc_3d[0]), float(loc_3d[1]), float(loc_3d[2])], dtype=np.float32
                        )
            
            return node_to_loc, node_to_loc_3d
    
    raise FileNotFoundError(f"PKL file not found for {subject}_{hemi}")


def find_overlapping_nodes(
    template_node_to_coords: dict[int, np.ndarray],
    subject_node_to_coords: dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find overlapping node IDs between template and subject.
    
    Returns:
        overlapping_node_ids: array of node IDs that exist in both
        template_coords: (N, D) coordinates for overlapping nodes in template
        subject_coords: (N, D) coordinates for overlapping nodes in subject
    """
    template_node_ids = set(template_node_to_coords.keys())
    subject_node_ids = set(subject_node_to_coords.keys())
    overlapping_node_ids = sorted(template_node_ids & subject_node_ids)
    
    if len(overlapping_node_ids) == 0:
        raise ValueError("No overlapping nodes found between template and subject")
    
    template_coords = np.array([template_node_to_coords[nid] for nid in overlapping_node_ids])
    subject_coords = np.array([subject_node_to_coords[nid] for nid in overlapping_node_ids])
    
    return np.array(overlapping_node_ids), template_coords, subject_coords


def thin_plate_spline_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    query_points: np.ndarray,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Apply Thin-Plate Spline (TPS) transformation using RBF interpolation.
    
    Args:
        source_points: (N, D) control points in source space
        target_points: (N, D) corresponding control points in target space
        query_points: (M, D) points to transform
        smoothing: smoothing parameter (0 = exact interpolation, >0 = smoothing)
    
    Returns:
        transformed_points: (M, D) transformed query points
    """
    if len(source_points) < 3:
        # Not enough points for TPS, use affine transformation
        from scipy.spatial.transform import Rotation
        
        # Compute affine transformation
        source_centroid = source_points.mean(axis=0)
        target_centroid = target_points.mean(axis=0)
        
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # Use SVD to find best rotation
        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply transformation
        query_centered = query_points - source_centroid
        transformed = query_centered @ R.T + target_centroid
        return transformed
    
    if HAS_RBF is False:
        # Fallback: use scipy.interpolate.griddata with linear interpolation
        from scipy.interpolate import griddata
        
        dim = source_points.shape[1]
        transformed_points = np.zeros_like(query_points)
        for d in range(dim):
            transformed_points[:, d] = griddata(
                source_points,
                target_points[:, d],
                query_points,
                method="linear",
                fill_value=np.nan,
            )
        # Fill NaN values with nearest neighbor interpolation
        nan_mask = np.isnan(transformed_points).any(axis=1)
        if nan_mask.any():
            from scipy.spatial.distance import cdist
            
            dists = cdist(query_points[nan_mask], source_points)
            nearest_idx = np.argmin(dists, axis=1)
            transformed_points[nan_mask] = target_points[nearest_idx]
        return transformed_points
    
    # Use RBF interpolation with thin-plate spline kernel
    # For 2D: r^2 * log(r), for 3D: r
    dim = source_points.shape[1]
    if dim == 2:
        kernel = "thin_plate_spline"  # r^2 * log(r)
    elif dim == 3:
        kernel = "linear"  # r (for 3D, linear is closer to TPS behavior)
    else:
        kernel = "thin_plate_spline"
    
    # Fit RBF interpolator for each dimension
    transformed_points = np.zeros_like(query_points)
    for d in range(dim):
        rbf = RBFInterpolator(
            source_points,
            target_points[:, d],
            kernel=kernel,
            smoothing=smoothing,
            epsilon=None,  # Auto-determine epsilon
        )
        transformed_points[:, d] = rbf(query_points)
    
    return transformed_points


def align_with_node_correspondence(
    template_mid_coords: np.ndarray,
    template_global_vertex_idx: np.ndarray,
    subject_mid_coords: np.ndarray,
    subject_global_vertex_idx: np.ndarray,
    template_subject: str,
    template_hemi: str,
    subject_subject: str,
    subject_hemi: str,
    repo_root: Path,
    tps_smoothing: float = 0.0,
) -> np.ndarray:
    """
    Align subject coordinates to template using node correspondence from PKL files.
    Uses Thin-Plate Spline (TPS) transformation.
    
    Args:
        template_mid_coords: (N_t, 3) template mid-surface coordinates
        template_global_vertex_idx: (N_t,) template global vertex indices (node IDs)
        subject_mid_coords: (N_s, 3) subject mid-surface coordinates
        subject_global_vertex_idx: (N_s,) subject global vertex indices (node IDs)
        template_subject: template subject name (e.g., "R1")
        template_hemi: template hemisphere (e.g., "lh")
        subject_subject: subject name (e.g., "S1")
        subject_hemi: subject hemisphere (e.g., "lh")
        repo_root: repository root path
        tps_smoothing: TPS smoothing parameter (0 = exact interpolation)
    
    Returns:
        aligned_subject_coords: (N_s, 3) aligned subject coordinates
    """
    # Load node mappings from PKL files
    template_node_to_loc, _ = load_pkl_node_mapping(template_subject, template_hemi, repo_root)
    subject_node_to_loc, _ = load_pkl_node_mapping(subject_subject, subject_hemi, repo_root)
    
    # Create mapping from global_vertex_idx to mid_coords index
    template_idx_to_coords = {
        int(gvid): template_mid_coords[i] for i, gvid in enumerate(template_global_vertex_idx)
    }
    subject_idx_to_coords = {
        int(gvid): subject_mid_coords[i] for i, gvid in enumerate(subject_global_vertex_idx)
    }
    
    # Find overlapping nodes (nodes that exist in both PKL files AND in both mid_coords)
    template_node_ids = set(template_node_to_loc.keys()) & set(template_idx_to_coords.keys())
    subject_node_ids = set(subject_node_to_loc.keys()) & set(subject_idx_to_coords.keys())
    overlapping_node_ids = sorted(template_node_ids & subject_node_ids)
    
    if len(overlapping_node_ids) < 3:
        raise ValueError(
            f"Not enough overlapping nodes ({len(overlapping_node_ids)}). Need at least 3 for alignment."
        )
    
    print(f"[align] Found {len(overlapping_node_ids)} overlapping nodes for alignment")
    
    # Extract coordinates for overlapping nodes
    template_overlap_coords = np.array([template_idx_to_coords[nid] for nid in overlapping_node_ids])
    subject_overlap_coords = np.array([subject_idx_to_coords[nid] for nid in overlapping_node_ids])
    
    print(f"[align] Using TPS transformation with {len(overlapping_node_ids)} control points")
    aligned_coords = thin_plate_spline_transform(
        source_points=subject_overlap_coords,
        target_points=template_overlap_coords,
        query_points=subject_mid_coords,
        smoothing=tps_smoothing,
    )
    
    return aligned_coords.astype(np.float32)
