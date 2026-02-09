from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick best (min final train_mse) TAG per hemi from R1_*_<TAG>.history.json")
    p.add_argument("--repo-root", type=str, default=".", help="Repo root (contains geognn_ot/).")
    p.add_argument(
        "--hemi",
        type=str,
        choices=["both", "lh", "rh"],
        default="both",
        help="Which hemisphere(s) to require. Default: both.",
    )
    p.add_argument(
        "--contains",
        type=str,
        default="sweepR1",
        help="Only consider tags containing this substring.",
    )
    p.add_argument(
        "--max-train-mse",
        type=float,
        default=0.01,
        help="Optional cutoff: ignore runs whose final train_mse is greater than this value.",
    )
    p.add_argument(
        "--module-name",
        type=str,
        default="",
        help="Filter by module name (e.g., 'intrinsic_normal_xyz'). If empty, picks best for each module combination separately.",
    )
    p.add_argument(
        "--print-exports",
        action="store_true",
        help="Print shell exports (TAG_LH=... etc) instead of a human message.",
    )
    return p.parse_args()


def load_final_train_mse(path: Path) -> float:
    hist = json.loads(path.read_text())
    if not isinstance(hist, list) or not hist:
        return float("nan")
    last = hist[-1]
    if not isinstance(last, dict):
        return float("nan")
    try:
        # Prefer new key (min-max tuning MSE), fallback to legacy key.
        if "train_mse_minmax" in last:
            return float(last["train_mse_minmax"])
        return float(last["train_mse"])
    except Exception:
        return float("nan")


TAG_RE = re.compile(
    r"(?:_fm(?P<fm>orig|xyz|both)_)?"
    r"t(?P<temp>[^_]+)_"
    r"lr(?P<lr>[^_]+)_"
    r"k(?P<k>\\d+)_"
    r"wd(?P<wd>[^_]+)_"
    r"clip(?P<clip>[^_]+)"
    r"(?:_si(?P<si>\\d+))?"
)


def _unsanitize_num(s: str) -> float:
    # inverse of sed: '.' -> 'p', '-' -> 'm'
    s = s.replace("m", "-").replace("p", ".")
    return float(s)


def parse_tag(tag: str) -> dict:
    m = TAG_RE.search(tag)
    if not m:
        return {}
    d = m.groupdict()
    out = {
        "temp": _unsanitize_num(d["temp"]),
        "lr": _unsanitize_num(d["lr"]),
        "cand_k_src": int(d["k"]),
        "weight_decay": _unsanitize_num(d["wd"]),
        "clip_norm": _unsanitize_num(d["clip"]),
    }
    return out


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)

    def _ckpt_dir(root: Path) -> Path:
        return root / "geognn_ot" / "results" / "checkpoints"

    ckpt_dir = _ckpt_dir(repo_root)
    if not ckpt_dir.exists():
        # Cluster-specific convenience: try common alternate roots (home vs coda).
        user = os.environ.get("USER", "")
        candidates: list[Path] = []
        if user:
            candidates.append(Path("/storage/coda1/p-nimam6/0") / user / "self_organization/transfer")
            candidates.append(Path("/storage/home/hcoda1/3") / user / "p-nimam6-0/self_organization/transfer")
        candidates.append(repo_root.resolve())

        for cand in candidates:
            d = _ckpt_dir(cand)
            if d.exists():
                repo_root = cand
                ckpt_dir = d
                break

    if not ckpt_dir.exists():
        raise SystemExit(f"Missing checkpoints dir: {ckpt_dir}")

    pat = re.compile(r"^R1_(lh|rh)_(.+)\.history\.json$")

    # Group by (hemi, module_name) -> best (mse, tag)
    best_by_module: dict[tuple[str, str], tuple[float, str]] = {}
    
    for p in ckpt_dir.rglob("R1_*_*.history.json"):
        m = pat.match(p.name)
        if not m:
            continue
        hemi, tag = m.group(1), m.group(2)
        if args.contains and args.contains not in tag:
            continue
        
        module_name = p.parent.name
        
        # Filter by module_name if specified
        if args.module_name and module_name != args.module_name:
            continue
        
        mse = load_final_train_mse(p)
        if not math.isfinite(mse):
            continue
        if mse > float(args.max_train_mse):
            continue
        
        key = (hemi, module_name)
        if key not in best_by_module or mse < best_by_module[key][0]:
            best_by_module[key] = (mse, tag)

    # If module_name specified, return best for that module
    # Otherwise, return best for each module separately
    if args.module_name:
        required = ["lh", "rh"] if args.hemi == "both" else [args.hemi]
        best: dict[str, tuple[float, str]] = {}
        for h in required:
            key = (h, args.module_name)
            if key in best_by_module:
                best[h] = best_by_module[key]
        
        if any(h not in best for h in required):
            missing = [h for h in required if h not in best]
            raise SystemExit(f"Could not find best tag for module '{args.module_name}' and hemi: {missing}. Try a different --contains or --module-name.")
        
        lh_mse, lh_tag = best.get("lh", (float("nan"), ""))
        rh_mse, rh_tag = best.get("rh", (float("nan"), ""))
        
        if args.print_exports:
            if args.hemi in ("both", "lh"):
                print(f"TAG_LH={lh_tag}")
                print(f"EVAL_TAG_LH={lh_tag}")
                print(f"BEST_LH_MSE={lh_mse}")
                hp = parse_tag(lh_tag)
                for k, v in hp.items():
                    print(f"BEST_LH_{k.upper()}={v}")
            if args.hemi in ("both", "rh"):
                print(f"TAG_RH={rh_tag}")
                print(f"EVAL_TAG_RH={rh_tag}")
                print(f"BEST_RH_MSE={rh_mse}")
                hp = parse_tag(rh_tag)
                for k, v in hp.items():
                    print(f"BEST_RH_{k.upper()}={v}")
        else:
            if args.hemi in ("both", "lh"):
                print(f"lh ({args.module_name}): {lh_tag} (final train_mse={lh_mse:.6f})")
            if args.hemi in ("both", "rh"):
                print(f"rh ({args.module_name}): {rh_tag} (final train_mse={rh_mse:.6f})")
    else:
        # Print best for each module combination
        modules = sorted(set(mod for _, mod in best_by_module.keys()))
        required = ["lh", "rh"] if args.hemi == "both" else [args.hemi]
        
        for mod in modules:
            best: dict[str, tuple[float, str]] = {}
            for h in required:
                key = (h, mod)
                if key in best_by_module:
                    best[h] = best_by_module[key]
            
            if not best:
                continue
            
            lh_mse, lh_tag = best.get("lh", (float("nan"), ""))
            rh_mse, rh_tag = best.get("rh", (float("nan"), ""))
            
            if args.print_exports:
                print(f"# Module: {mod}")
                if args.hemi in ("both", "lh"):
                    print(f"TAG_LH_{mod.replace('_', '').upper()}={lh_tag}")
                    print(f"EVAL_TAG_LH_{mod.replace('_', '').upper()}={lh_tag}")
                    print(f"BEST_LH_MSE_{mod.replace('_', '').upper()}={lh_mse}")
                if args.hemi in ("both", "rh"):
                    print(f"TAG_RH_{mod.replace('_', '').upper()}={rh_tag}")
                    print(f"EVAL_TAG_RH_{mod.replace('_', '').upper()}={rh_tag}")
                    print(f"BEST_RH_MSE_{mod.replace('_', '').upper()}={rh_mse}")
            else:
                print(f"\nModule: {mod}")
                if args.hemi in ("both", "lh"):
                    print(f"  lh: {lh_tag} (final train_mse={lh_mse:.6f})")
                if args.hemi in ("both", "rh"):
                    print(f"  rh: {rh_tag} (final train_mse={rh_mse:.6f})")


if __name__ == "__main__":
    main()

