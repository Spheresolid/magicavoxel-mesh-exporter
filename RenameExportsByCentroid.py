import os
import glob
import shutil
import math
import argparse
import datetime
import json
import uuid
import numpy as np
import trimesh
from Vox200Parser import Vox200Parser

def sanitize_filename(name):
    return ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in (name or "")).strip()

BASE_DIR = os.path.dirname(__file__)
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

parser_arg = argparse.ArgumentParser(description="Rename exported .obj files to match parser intent by global assignment.")
parser_arg.add_argument("--commit", action="store_true", help="Perform renames (default is dry-run). Backups are created for overwritten files.")
parser_arg.add_argument("--vox", help="Optional: limit to a single .vox basename (no extension).")
parser_arg.add_argument("--count-weight", type=float, default=0.5, help="Weight for voxel-count penalty relative to centroid distance (default 0.5)")
parser_arg.add_argument("--iou-weight", type=float, default=1.0, help="Weight for AABB IoU penalty (1 - IoU) (default 1.0)")
args = parser_arg.parse_args()

def backup_path(path):
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    base = os.path.basename(path)
    return os.path.join(os.path.dirname(path), f"{base}.bak.{ts}")

def read_exportlog_counts(export_dir):
    """Parse ExportLog.txt if present to get declared voxel counts for exported files."""
    out = {}
    p = os.path.join(export_dir, "ExportLog.txt")
    if not os.path.exists(p):
        return out
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("exported:"):
                    try:
                        parts = line.split("Exported:")[1].strip()
                        if " voxels" in parts:
                            before, _ = parts.rsplit(" voxels", 1)
                            if " as " in before:
                                fullpath, after = before.split(" as ", 1)
                                filename = os.path.basename(fullpath.strip())
                                try:
                                    count = int(parts.strip().split()[-2])
                                except Exception:
                                    count = None
                                out[filename] = count
                    except Exception:
                        continue
    except Exception:
        pass
    return out

def try_hungarian(cost_matrix):
    """Try to use scipy's Hungarian; return row_ind, col_ind or None if not available."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost_matrix)
        return r.tolist(), c.tolist()
    except Exception:
        return None

def greedy_with_pairwise_improvement(cost):
    """
    Pure-Python fallback:
    - greedy initial assignment by picking minimal cost pairs
    - then try pairwise swaps to reduce total cost until no improvement
    Returns two lists: row_idx -> col_idx mapping
    """
    n_rows, n_cols = cost.shape
    rows = list(range(n_rows))
    cols = list(range(n_cols))
    assigned_cols = set()
    assignment = [-1] * n_rows

    # Greedy: for each row, pick nearest available column (smallest cost)
    for r in rows:
        # sort columns by cost for this row
        col_order = sorted(cols, key=lambda c: cost[r, c])
        chosen = None
        for c in col_order:
            if c not in assigned_cols:
                chosen = c
                break
        if chosen is None:
            # no available columns -> pick any
            for c in cols:
                if c not in assigned_cols:
                    chosen = c
                    break
        if chosen is not None:
            assignment[r] = chosen
            assigned_cols.add(chosen)

    # If unassigned rows (more rows than cols), leave -1, but we'll fill with dummy cols
    # Convert to full mapping by filling missing with unused cols or -1
    unused_cols = [c for c in cols if c not in assigned_cols]
    for i, a in enumerate(assignment):
        if a == -1:
            if unused_cols:
                assignment[i] = unused_cols.pop(0)
            else:
                assignment[i] = None

    # Pairwise improvement: try swapping assigned columns between any two rows
    improved = True
    def total_cost(assign):
        s = 0.0
        for i, c in enumerate(assign):
            if c is None:
                s += 1e6
            else:
                s += cost[i, c]
        return s

    current_cost = total_cost(assignment)
    while improved:
        improved = False
        n = len(assignment)
        for i in range(n):
            for j in range(i+1, n):
                ai = assignment[i]
                aj = assignment[j]
                if ai is None or aj is None:
                    continue
                # try swapping
                new_assign = assignment[:]
                new_assign[i], new_assign[j] = aj, ai
                new_cost = total_cost(new_assign)
                if new_cost + 1e-12 < current_cost:
                    assignment = new_assign
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    # convert to lists
    row_idx = [i for i in range(len(assignment))]
    col_idx = [assignment