"""
run_batch.py — parameter sweep runner for ppglm.py
====================================================

Define sweeps as  param_name -> [start, stop, step]  in SWEEP_PARAMS below,
then choose a MODE:

    "grid"   — cartesian product of all ranges (all combinations)
    "each"   — vary one parameter at a time; all others stay at DEFAULT_PARAMS

Results land in  OUTPUT_DIR/<timestamp>/  as one JSON file per run, plus a
summary CSV at the end.

Usage:
    python ppglm/run_batch.py                  # uses config below
    python ppglm/run_batch.py --dry_run        # print commands only
    python ppglm/run_batch.py --workers 4      # parallel jobs (default: 1)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────

# Script to sweep over (relative to this file's directory)
TARGET_SCRIPT = Path(__file__).parent / "ppglm.py"

# Output directory for results
OUTPUT_DIR = Path(__file__).parent.parent / "batch_results"

# Default values (used in "each" mode for params not being swept)
DEFAULT_PARAMS: Dict[str, Any] = {
    "seed": 42,
    "n_sims": 10,
    "global_syn_weight": 3.5,
    "pr": 0.1,
    "input_frequency_hz": 80.0,
}

# Sweep ranges: param_name -> [start, stop, step]
# Set to None or remove to skip a param.
SWEEP_PARAMS: Dict[str, List[float]] = {
    "global_syn_weight": [1.0, 6.0, 0.5],   # e.g. 1.0, 1.5, 2.0 … 6.0
    # "pr":                [0.05, 0.5, 0.05],
    # "input_frequency_hz":[20.0, 120.0, 20.0],
    # "n_sims":            [5, 50, 5],
}

# "grid"  → cartesian product of all sweep ranges
# "each"  → sweep one param at a time, others held at DEFAULT_PARAMS
MODE: str = "grid"

# ─────────────────────────────────────────────────────────────────────────────


def _arange_inclusive(start: float, stop: float, step: float) -> List[float]:
    """np.arange with inclusive stop, rounded to avoid float noise."""
    vals = np.arange(start, stop + step * 0.5, step)
    # Round to the number of decimal places in step
    decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    return [round(float(v), decimals) for v in vals]


def build_param_grid(sweep: Dict[str, List[float]], mode: str) -> Iterator[Dict[str, Any]]:
    """Yield param dicts for every run."""
    ranges = {k: _arange_inclusive(*v) for k, v in sweep.items()}

    if mode == "grid":
        keys = list(ranges.keys())
        for combo in itertools.product(*[ranges[k] for k in keys]):
            params = dict(DEFAULT_PARAMS)
            params.update(dict(zip(keys, combo)))
            yield params

    elif mode == "each":
        for param, values in ranges.items():
            for val in values:
                params = dict(DEFAULT_PARAMS)
                params[param] = val
                yield params

    else:
        raise ValueError(f"Unknown MODE={mode!r}. Choose 'grid' or 'each'.")


def params_to_cmd(params: Dict[str, Any]) -> List[str]:
    """Convert a param dict to a subprocess command list."""
    cmd = [sys.executable, str(TARGET_SCRIPT)]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def run_one(
    run_id: int,
    params: Dict[str, Any],
    out_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single ppglm.py invocation and return a result record."""
    cmd = params_to_cmd(params)
    result_path = out_dir / f"run_{run_id:04d}.json"

    if dry_run:
        print(f"[dry] {' '.join(cmd)}")
        return {"run_id": run_id, **params, "status": "dry_run", "returncode": None}

    print(f"[run {run_id:04d}] {' '.join(cmd)}")
    t0 = datetime.now()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (datetime.now() - t0).total_seconds()

    record: Dict[str, Any] = {
        "run_id": run_id,
        **params,
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 2),
        "stdout": proc.stdout[-2000:] if proc.stdout else "",
        "stderr": proc.stderr[-2000:] if proc.stderr else "",
    }

    result_path.write_text(json.dumps(record, indent=2))

    if proc.returncode != 0:
        print(f"  ✗ run {run_id:04d} failed (rc={proc.returncode})")
        print(proc.stderr[-400:])
    else:
        print(f"  ✓ run {run_id:04d} done in {elapsed:.1f}s")

    return record


def write_summary(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Write a flat CSV summary of all runs."""
    if not records:
        return
    summary_path = out_dir / "summary.csv"
    # Collect all keys (preserve insertion order, params first)
    keys = list(dict.fromkeys(k for r in records for k in r if k not in ("stdout", "stderr")))
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSummary written → {summary_path}")


def main() -> None:
    cli = argparse.ArgumentParser(description="Batch parameter sweep for ppglm.py")
    cli.add_argument("--dry_run", action="store_true", help="Print commands without running them")
    cli.add_argument("--workers", type=int, default=1, help="Parallel worker processes (default: 1)")
    args = cli.parse_args()

    # Build the run list
    param_list = list(build_param_grid(SWEEP_PARAMS, MODE))
    n_runs = len(param_list)
    print(f"Mode: {MODE!r} | Runs planned: {n_runs} | Workers: {args.workers}")
    if not param_list:
        print("No parameter combinations generated — check SWEEP_PARAMS.")
        return

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the sweep config alongside results
    config = {"mode": MODE, "sweep_params": SWEEP_PARAMS, "defaults": DEFAULT_PARAMS}
    (out_dir / "sweep_config.json").write_text(json.dumps(config, indent=2))

    records: List[Dict[str, Any]] = []

    if args.workers > 1 and not args.dry_run:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, i, p, out_dir, args.dry_run): i
                for i, p in enumerate(param_list)
            }
            for fut in as_completed(futures):
                records.append(fut.result())
    else:
        for i, params in enumerate(param_list):
            records.append(run_one(i, params, out_dir, args.dry_run))

    if not args.dry_run:
        records.sort(key=lambda r: r["run_id"])
        write_summary(records, out_dir)
        ok = sum(1 for r in records if r["status"] == "ok")
        print(f"\nDone: {ok}/{n_runs} runs succeeded.")


if __name__ == "__main__":
    main()
