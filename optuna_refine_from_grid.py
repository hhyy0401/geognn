from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import optuna


def load_candidates(grid_path: Path) -> List[Dict[str, Any]]:
    if not grid_path.exists():
        raise FileNotFoundError(f"grid path not found: {grid_path}")
    
    files = []
    if grid_path.is_dir():
        # Load all candidates_*.json in the directory
        files = list(grid_path.glob("candidates_*.json"))
        # Also check for the legacy candidates.json
        legacy = grid_path / "candidates.json"
        if legacy.exists():
            files.append(legacy)
    else:
        files = [grid_path]

    if not files:
        raise FileNotFoundError(f"No candidate files found in {grid_path}")

    out = []
    for f_path in files:
        with open(f_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[warn] could not parse {f_path}, skipping.")
                continue
        if isinstance(data, list):
            out.extend(data)
        elif isinstance(data, dict):
            out.append(data)
    
    if not out:
        raise ValueError(f"no candidates found in {grid_path}")
    return out


def bounds_from_candidates(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    def get(name: str, default=None):
        vals = [c.get(name, default) for c in cands if c.get(name, default) is not None]
        return vals

    temp = [float(v) for v in get("temp")]
    lr = [float(v) for v in get("lr")]
    wd = [float(v) for v in get("weight_decay")]
    clip = [float(v) for v in get("clip_norm")]
    k = sorted({int(v) for v in get("cand_k_src")})
    topo = [float(v) for v in get("topo_lambda", 0.0)]
    
    def mm(xs: List[float], pad: float = 0.0) -> tuple[float, float]:
        if not xs: return 0.0, 0.0
        lo, hi = min(xs), max(xs)
        if pad > 0:
            span = max(1e-12, hi - lo)
            lo -= pad * span
            hi += pad * span
        return lo, hi

    bounds = {
        "temp": mm(temp, pad=0.15),
        "lr": mm(lr, pad=0.15),
        "weight_decay": mm(wd, pad=0.15),
        "clip_norm": mm(clip, pad=0.15),
        "cand_k_src": k,
    }

    dp_vals = get("dist_penalty_lambda")
    if dp_vals:
        dp_vals_float = [float(v) for v in dp_vals]
        bounds["dist_penalty_lambda"] = float(sum(dp_vals_float) / max(1, len(dp_vals_float)))

    topo_vals = get("topo_lambda")
    if topo_vals:
         topo_vals_float = [float(v) for v in topo_vals]
         bounds["topo_lambda"] = mm(topo_vals_float, pad=0.15)
    
    return bounds


def read_final_train_mse(history_json: Path) -> float:
    with open(history_json, "r") as f:
        hist = json.load(f)
    # train.py writes a LIST of dicts:
    #   [{"epoch":..., "train_mse_minmax": ..., "lr": ...}, ...]
    # Older versions may have {"train": [...]} or "train_mse".
    if isinstance(hist, list):
        train_hist = hist
    elif isinstance(hist, dict):
        train_hist = hist.get("train", [])
    else:
        train_hist = []
    if not train_hist:
        raise ValueError(f"history missing train curve: {history_json}")
    last = train_hist[-1]
    if not isinstance(last, dict):
        raise ValueError(f"history last entry is not a dict: {history_json}")
    if "train_mse_zscore" in last:
        return float(last["train_mse_zscore"])
    if "train_mse_minmax" in last:
        return float(last["train_mse_minmax"])
    if "train_mse" in last:
        return float(last["train_mse"])
    # fallback: pick any numeric in last
    for _, v in last.items():
        if isinstance(v, (int, float)):
            return float(v)
    raise ValueError(f"cannot parse final train mse: {history_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna HPO constrained by grid-search candidate set (R1 only).")
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--hemi", type=str, choices=["lh", "rh"], required=True)
    ap.add_argument("--module-name", type=str, default="none")
    ap.add_argument("--grid-json", type=str, default="", help="Path to results/grid/<hemi>/candidates.json")
    ap.add_argument("--optuna-prefix", type=str, default="optunaR1")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sinkhorn-iters", type=int, default=20)
    ap.add_argument("--cand-method", type=str, default="score_topk")
    ap.add_argument("--cand-refresh-every", type=int, default=1)
    ap.add_argument("--score-topk-src-chunk", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--gat-heads", type=int, default=4)
    ap.add_argument("--gat-dropout", type=float, default=0.0)
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--mlp-dropout", type=float, default=0.0)
    ap.add_argument("--use-intrinsic", action="store_true")
    ap.add_argument("--use-normal", action="store_true")
    ap.add_argument("--use-xyz", action="store_true")
    ap.add_argument("--use-distance", action="store_true")
    ap.add_argument("--patience", type=int, default=100)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    grid_json = Path(args.grid_json) if args.grid_json else (repo_root / "geognn_ot" / "results" / "grid" / args.module_name / args.hemi / "candidates.json")
    cands = load_candidates(grid_json)
    b = bounds_from_candidates(cands)

    ckpt_dir = repo_root / "geognn_ot" / "results" / "checkpoints" / args.module_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    params_dir = repo_root / "geognn_ot" / "results" / "params" / args.module_name
    params_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        temp = trial.suggest_float("temp", b["temp"][0], b["temp"][1])
        lr = trial.suggest_float("lr", b["lr"][0], b["lr"][1], log=True)
        wd_lo, wd_hi = b["weight_decay"][0], b["weight_decay"][1]
        if wd_hi <= 0 or wd_lo == wd_hi:
            wd = float(max(0.0, wd_hi))
        elif wd_lo <= 0:
            wd = trial.suggest_float("weight_decay", 0.0, float(wd_hi), log=False)
        else:
            wd = trial.suggest_float("weight_decay", float(wd_lo), float(wd_hi), log=True)
        clip = trial.suggest_float("clip_norm", b["clip_norm"][0], b["clip_norm"][1])
        k = trial.suggest_categorical("cand_k_src", b["cand_k_src"])
        tag_parts = [
            f"{args.optuna_prefix}",
            f"t{temp:.4g}",
            f"lr{lr:.4g}",
            f"k{k}",
            f"wd{wd:.4g}",
            f"clip{clip:.4g}",
        ]

        dp = None
        if "dist_penalty_lambda" in b:
            dp = float(b["dist_penalty_lambda"])
            dp_s = str(dp).replace(".", "p")
            tag_parts.append(f"dp{dp_s}")

        topo = None
        if "topo_lambda" in b:
             topo = trial.suggest_float("topo_lambda", b["topo_lambda"][0], b["topo_lambda"][1])
             topo_s = str(round(topo, 4)).replace(".", "p")
             tag_parts.append(f"topo{topo_s}")
        
        tag_parts.append(f"s{args.seed}")
        tag_parts.append(f"tr{trial.number}")
        tag = "_".join(tag_parts)
        ckpt_out = ckpt_dir / f"R1_{args.hemi}_{tag}.pt"

        cmd = [
            "python",
            "-u",
            "-m",
            "geognn_ot.train",
            "train",
            "--repo-root",
            str(repo_root),
            "--hemi",
            args.hemi,
            "--train-subjects",
            "R1",
            "--out-dir",
            str(ckpt_dir),
            "--ckpt-out",
            str(ckpt_out),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(lr),
            "--weight-decay",
            str(wd),
            "--scheduler",
            "cosine",
            "--clip-norm",
            str(clip),
            "--device",
            args.device,
            "--hidden-dim",
            str(args.hidden_dim),
            "--gat-heads",
            str(args.gat_heads),
            "--gat-dropout",
            str(args.gat_dropout),
            "--sinkhorn-iters",
            str(args.sinkhorn_iters),
            "--temp",
            str(temp),
            "--mlp-hidden",
            str(args.mlp_hidden),
            "--mlp-dropout",
            str(args.mlp_dropout),
            "--cand-k-src",
            str(k),
            "--cand-method",
            args.cand_method,
            "--cand-refresh-every",
            str(args.cand_refresh_every),
            "--score-topk-src-chunk",
            str(args.score_topk_src_chunk),
            "--seed",
            str(args.seed),
            "--patience",
            str(args.patience),
            "--log-every",
            "50",
        ]
        
        if dp is not None:
            cmd.extend(["--dist-penalty-lambda", str(dp)])
        if topo is not None:
             cmd.extend(["--topo-lambda", str(topo)])

        if args.use_intrinsic:
            cmd.append("--use-intrinsic")
        if args.use_normal:
            cmd.append("--use-normal")
        if args.use_xyz:
            cmd.append("--use-xyz")
        if args.use_distance:
            cmd.append("--use-distance")

        subprocess.run(cmd, check=True)

        hist_path = ckpt_out.with_suffix(".history.json")
        mse = read_final_train_mse(hist_path)
        trial.set_user_attr("tag", tag)
        trial.set_user_attr("ckpt", str(ckpt_out))
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    best_tag = best.user_attrs.get("tag", "")
    best_ckpt = best.user_attrs.get("ckpt", "")
    out = {
        "hemi": args.hemi,
        "module": args.module_name,
        "best_tag": best_tag,
        "best_ckpt": best_ckpt,
        "best_train_mse": float(best.value),
        "best_params": best.params,
        "grid_json": str(grid_json),
    }

    best_path = params_dir / f"best_{args.hemi}.json"
    with open(best_path, "w") as f:
        json.dump(out, f, indent=2)

    optuna_log = params_dir / f"optuna_{args.hemi}_trials.json"
    with open(optuna_log, "w") as f:
        json.dump(
            [
                {"number": t.number, "value": t.value, "params": t.params, "tag": t.user_attrs.get("tag", "")}
                for t in study.trials
                if t.value is not None
            ],
            f,
            indent=2,
        )

    print(f"[ok] wrote {best_path} (best_train_mse={best.value})")


if __name__ == "__main__":
    main()
