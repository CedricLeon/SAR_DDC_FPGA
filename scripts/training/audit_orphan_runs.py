#!/usr/bin/env python3
"""audit_orphan_runs.py — find local Hydra run directories that no W&B run points at.

``logs/`` accumulates every training attempt (crashed runs, deprecated sweeps, debug runs), while
W&B only keeps the ones that were actually logged and kept. This lists local run dirs that are
**orphans**: not referenced by any W&B run's ``paths.output_dir``, and not referenced by any
compiled FPGA model's ``manifest.json::original_run_dir`` (those are needed to re-quantize).

Reports sizes so the big wins are obvious, and writes the orphan list to a file. Deletion is a
separate, explicit step — this script never deletes unless ``--delete`` is passed, and even then
it prints the total and requires ``--yes``.

    python scripts/training/audit_orphan_runs.py                      # audit, live W&B query
    python scripts/training/audit_orphan_runs.py --source csv         # use the exported CSV
    python scripts/training/audit_orphan_runs.py --delete --yes       # after reviewing the list
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Set

import rootutils

REPO_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

ENTITY, PROJECT = "cedric-leonard", "SAR_DDC_FPGA"
LOGS_DIR = REPO_ROOT / "logs"
COMPILED_DIR = REPO_ROOT / "results" / "fpga" / "compiled_models"
WANDB_CSV = REPO_ROOT / "notebooks" / "SAR_DDC_FPGA_all_runs_WandB.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "orphan_runs.txt"


def _canon(path) -> Path:
    """Canonical absolute path, symlinks resolved.

    Essential: W&B stores ``/mnt/.../DDC_FPGA/logs/...`` while the same tree may be reached
    through a git-worktree symlink. Comparing unresolved strings makes every live run look
    like an orphan.
    """
    return Path(path).resolve()


def _short(path: Path) -> str:
    """Path from ``logs/`` down, for readable listings."""
    parts = path.parts
    return str(Path(*parts[parts.index("logs") :])) if "logs" in parts else str(path)


def wandb_run_dirs(source: str) -> Set[Path]:
    """Every ``paths.output_dir`` known to W&B, live or from the exported CSV."""
    if source == "csv":
        import pandas as pd

        if not WANDB_CSV.exists():
            raise SystemExit(f"CSV not found: {WANDB_CSV} (run notebooks/fetch_wandb_runs.py)")
        col = pd.read_csv(WANDB_CSV)["run_dir"].dropna()
        print(f"[audit] {len(col)} run dirs from {WANDB_CSV.name}")
        return {_canon(p) for p in col}

    import wandb

    runs = wandb.Api().runs(f"{ENTITY}/{PROJECT}")
    dirs = set()
    for run in runs:
        out = run.config.get("paths", {}).get("output_dir")
        if out:
            dirs.add(_canon(out))
    print(f"[audit] {len(dirs)} run dirs from W&B ({ENTITY}/{PROJECT}, {len(runs)} runs)")
    return dirs


def compiled_run_dirs() -> Set[Path]:
    """``original_run_dir`` of every compiled FPGA model — never delete these."""
    dirs = set()
    for man in sorted(COMPILED_DIR.glob("*/manifest.json")):
        run_dir = json.loads(man.read_text()).get("original_run_dir")
        if run_dir:
            dirs.add(_canon(run_dir))
    print(f"[audit] {len(dirs)} run dirs referenced by compiled FPGA models")
    return dirs


def local_run_dirs() -> List[Path]:
    """Hydra output dirs on disk: ``**/runs/<timestamp>`` and ``**/multiruns/<timestamp>/<idx>``."""
    found = []
    for parent in LOGS_DIR.rglob("runs"):
        found += [_canon(p) for p in parent.iterdir() if p.is_dir()]
    for parent in LOGS_DIR.rglob("multiruns"):
        for stamp in (p for p in parent.iterdir() if p.is_dir()):
            children = [p for p in stamp.iterdir() if p.is_dir() and p.name.isdigit()]
            found += [_canon(p) for p in (children if children else [stamp])]
    return sorted(set(found))


def dir_size(path: Path) -> int:
    """Bytes on disk (``du -sb``; far faster than walking in Python for GB-scale trees)."""
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", choices=["wandb", "csv"], default="wandb")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--delete", action="store_true", help="delete the orphans (needs --yes)")
    ap.add_argument("--yes", action="store_true", help="confirm deletion")
    args = ap.parse_args()

    keep = wandb_run_dirs(args.source) | compiled_run_dirs()
    local = local_run_dirs()
    orphans = [p for p in local if p not in keep]

    # Safety: if path matching breaks (unresolved symlinks, a moved logs/ root, a renamed W&B
    # config key), every live run silently looks like an orphan. Most known run dirs still exist
    # on disk, so a near-zero hit rate means the comparison is broken, not that the runs are gone.
    matched = len(local) - len(orphans)
    if matched < 0.5 * len(keep):
        raise SystemExit(
            f"refusing to continue: only {matched} of {len(keep)} known run dirs were found "
            f"among {len(local)} local dirs. Path matching is broken — compare a W&B "
            f"paths.output_dir value against the local layout before trusting this list."
        )
    print(
        f"[audit] {len(local)} local run dirs, {len(local) - len(orphans)} known, "
        f"{len(orphans)} orphans"
    )

    sizes = {p: dir_size(p) for p in orphans}
    total = sum(sizes.values())
    kept_total = sum(dir_size(p) for p in local if p in keep)

    print(f"\n{'orphan run dir':<80s}{'size':>10s}")
    for p, s in sorted(sizes.items(), key=lambda kv: -kv[1])[:25]:
        # Not relative_to(REPO_ROOT): logs/ may be reached through a worktree symlink, so the
        # resolved path can sit outside the repo root.
        print(f"{_short(p):<80s}{s / 2**30:>9.2f}G")
    if len(sizes) > 25:
        print(f"... and {len(sizes) - 25} more")
    print(f"\n  orphans : {total / 2**30:8.1f} GB  ({len(orphans)} dirs)")
    print(f"  kept    : {kept_total / 2**30:8.1f} GB  ({len(local) - len(orphans)} dirs)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(p) for p in sorted(orphans)) + "\n")
    print(f"  list -> {args.out}")

    if not args.delete:
        print("\n  (dry run — pass --delete --yes to remove the dirs listed above)")
        return
    if not args.yes:
        raise SystemExit("--delete requires --yes")
    for p in orphans:
        shutil.rmtree(p)
    print(f"\n  deleted {len(orphans)} dirs, freed {total / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
