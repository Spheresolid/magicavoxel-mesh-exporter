# -*- coding: utf-8 -*-
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
import traceback
import re
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
                if line.lower().startswith("exported"):
                    try:
                        # Find the filename and the voxel count
                        parts = line.split("as", 1)[1].strip()
                        filename = parts.split(" (source=", 1)[0].split()[0]
                        count_match = re.search(r'\((\d+)\s+voxels\)', line)
                        count = int(count_match.group(1)) if count_match else 0
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
    Pure-Python fallback for assignment problem.
    """
    n_rows, n_cols = cost.shape
    rows = list(range(n_rows))
    cols = list(range(n_cols))
    assigned_cols = set()
    assignment = [-1] * n_rows

    # Greedy: for each row, pick nearest available column (smallest cost)
    for r in rows:
        col_order = sorted(cols, key=lambda c: cost[r, c])
        chosen = None
        for c in col_order:
            if c not in assigned_cols:
                chosen = c
                break
        if chosen is not None:
            assignment[r] = chosen
            assigned_cols.add(chosen)
            
    unused_cols = [c for c in cols if c not in assigned_cols]
    for i, a in enumerate(assignment):
        if a == -1:
            if unused_cols:
                assignment[i] = unused_cols.pop(0)
            else:
                assignment[i] = None

    def total_cost(assign):
        s = 0.0
        for i, c in enumerate(assign):
            if c is None:
                s += 1e6
            else:
                s += cost[i, c]
        return s

    # Pairwise improvement
    current_cost = total_cost(assignment)
    improved = True
    while improved:
        improved = False
        n = len(assignment)
        for i in range(n):
            for j in range(i+1, n):
                ai = assignment[i]
                aj = assignment[j]
                if ai is None or aj is None:
                    continue
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

    row_idx = [i for i in range(len(assignment))]
    col_idx = [assignment[i] if assignment[i] is not None else -1 for i in range(len(assignment))]
    return row_idx, col_idx

def aabb_iou(min_a, max_a, min_b, max_b):
    # min_*/max_* are numpy arrays
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]
    vol_a = np.prod(np.maximum(max_a - min_a, 0.0))
    vol_b = np.prod(np.maximum(max_b - min_b, 0.0))
    union = vol_a + vol_b - inter_vol
    if union <= 0.0:
        return 0.0
    return float(inter_vol / union)

# --- NAME CLAIMING FUNCTION (patched) ---
def find_canonical_name(raw_key, parser, current_centroid, current_count, max_dist_norm, max_count_norm):
    """
    Enhanced logic:
    - Prefer parser's explicit non-generic name when it's reasonable (not suspicious).
    - If the parser-assigned name is non-generic but the part is suspiciously small, allow claiming.
    - Otherwise fall back to scanning parser.raw_part_name_map for nearby descriptive names.
    """
    # Parameters
    SUSPECT_VOXEL_THRESHOLD = 10  # parts smaller than this are suspicious even if named

    # Prefer the parser's explicit layer-name assignment when available
    current_name = parser.layer_name_map.get(raw_key)
    if current_name:
        # If assigned name is non-generic and the part is reasonably large, trust it
        if not str(current_name).startswith("Part_") and (current_count is not None and current_count >= SUSPECT_VOXEL_THRESHOLD):
            return current_name
        # If assigned name is generic, proceed to claim logic below
        # If assigned name is non-generic but the part is suspiciously small, allow claiming
    else:
        current_name = raw_key

    # Extract numeric index from raw_key (e.g., 'Part_12' -> 12). If not found, bail out.
    m_raw = re.search(r'Part_(\d+)', raw_key)
    if not m_raw:
        return current_name
    try:
        raw_idx = int(m_raw.group(1))
    except Exception:
        return current_name

    # Scan helper/raw part name map for clean names with nearby indices
    best_candidate = None
    best_score = None
    for helper_key, helper_name in parser.raw_part_name_map.items():
        if not helper_name:
            continue
        # Skip generic helper names
        if str(helper_name).startswith("Part_"):
            continue
        m_helper = re.search(r'Part_(\d+)', helper_key)
        if not m_helper:
            continue
        try:
            helper_idx = int(m_helper.group(1))
        except Exception:
            continue

        # distance metric: prefer smaller index distance and larger helper part count (if available)
        idx_dist = abs(helper_idx - raw_idx)
        helper_count = None
        try:
            helper_count = len(parser.voxels_by_layer.get(helper_key, []))
        except Exception:
            helper_count = None

        # score: lower is better; favor closer index and larger count
        score = idx_dist - (0.01 * (helper_count or 0))
        if best_score is None or score < best_score:
            best_score = score
            best_candidate = (helper_key, helper_name, helper_count, idx_dist)

    # If we found a candidate within a small index window, prefer it
    if best_candidate:
        helper_key, helper_name, helper_count, idx_dist = best_candidate
        # Only claim if close enough (index within 3) OR helper count is much larger than current
        if idx_dist <= 3 or (helper_count is not None and current_count is not None and helper_count >= 4 * max(1, current_count)):
            return helper_name

    # Nothing better found — return the original (possibly suspicious) name
    return current_name
# --- END NAME CLAIMING PATCH ---

# Main
vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]
if args.vox:
    target = f"{args.vox}.vox"
    if target in vox_files:
        vox_files = [target]
    else:
        print(f"Specified vox not found: {args.vox}")
        raise SystemExit(1)

if not vox_files:
    print("No .vox files found.")
    raise SystemExit(0)

for vox_file in vox_files:
    try:
        vox_path = os.path.join(BASE_DIR, vox_file)
        vox_base = os.path.splitext(vox_file)[0]
        print(f"Processing: {vox_file}")

        parser = Vox200Parser(vox_path).parse()
        voxels_by_layer = parser.voxels_by_layer         # dict 'Part_X' -> list of Voxels
        name_map = parser.layer_name_map                 # 'Part_X' -> intended name (for non-empty parts)
        model_keys = [k for k in voxels_by_layer.keys() if len(voxels_by_layer[k]) > 0]
        model_centroids = {}
        model_counts = {}
        for raw_key in model_keys:
            voxels = voxels_by_layer[raw_key]
            pts = np.array([[v.x, v.y, v.z] for v in voxels], dtype=float)
            model_centroids[raw_key] = pts.mean(axis=0)
            model_counts[raw_key] = len(voxels)

        exported_dir = os.path.join(EXPORT_ROOT, vox_base)
        exported_paths = sorted(glob.glob(os.path.join(exported_dir, "*.obj")))
        
        # --- ROBUST REPORTING CHECK ---
        report_path = os.path.join(REPORTS_DIR, f"RenamingReport_{vox_base}.txt")
        manifest_path = os.path.join(REPORTS_DIR, f"RenamingManifest_{vox_base}.json")

        if not exported_paths:
            print(f"  No exported .obj files found in {exported_dir}, skipping assignment.")
            manifest = {"vox": vox_file, "timestamp": datetime.datetime.now().isoformat(), "error": "No exported OBJ files found for assignment.", "renames": []}
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)
            with open(report_path, "w", encoding="utf-8") as rep:
                rep.write(f"RenamingReport (skipped) for: {vox_file}\n\n")
                rep.write(f"SKIPPED: No exported .obj files found in {exported_dir}.\n")
            
            print(f"  Report written: {report_path} (SKIPPED)")
            print(f"  Manifest written: {manifest_path} (SKIPPED)")
            continue
        # --- END ROBUST REPORTING CHECK ---

        exported_centroids = {}
        exported_mesh_counts = {}
        file_aabbs = {}
        # Geometry data extraction must be inside a try-except for safety
        for p in exported_paths:
            try:
                m = trimesh.load(p, force='mesh')
                c = np.array(m.centroid if hasattr(m, "centroid") else np.array(m.bounds).mean(axis=0), dtype=float)
                exported_centroids[p] = c
                file_aabbs[p] = (m.bounds[0].astype(float), m.bounds[1].astype(float))
            except Exception as e:
                print(f"  WARNING: Failed to process OBJ {os.path.basename(p)}: {e}")
                exported_centroids[p] = np.array([math.inf, math.inf, math.inf], dtype=float)
                file_aabbs[p] = (np.array([math.inf, math.inf, math.inf], dtype=float), np.array([-math.inf, -math.inf, -math.inf], dtype=float))

        exportlog_counts = read_exportlog_counts(exported_dir)
        for p in exported_paths:
            fname = os.path.basename(p)
            ccount = exportlog_counts.get(fname)
            if ccount is None:
                try:
                    m = trimesh.load(p, force='mesh')
                    est_vox = max(1, int(round(m.faces.shape[0] / 12.0))) if hasattr(m, "faces") else 0
                except Exception:
                    est_vox = 0
                exported_mesh_counts[p] = est_vox
            else:
                exported_mesh_counts[p] = ccount if ccount is not None else 0

        # compute model AABBs
        model_aabbs = {}
        for mk in model_keys:
            voxels = voxels_by_layer[mk]
            pts = np.array([[v.x, v.y, v.z] for v in voxels], dtype=float)
            model_aabbs[mk] = (pts.min(axis=0), pts.max(axis=0))

        # Build cost matrix rows = models, cols = exported files
        models = model_keys
        files = exported_paths
        R = len(models)
        C = len(files)
        N = max(R, C)
        cost = np.full((N, N), fill_value=1e6, dtype=float)

        # compute raw distances and count diffs for normalization
        dists = []
        count_diffs = []
        for i, mk in enumerate(models):
            for j, fp in enumerate(files):
                dist = np.linalg.norm(model_centroids[mk] - exported_centroids[fp])
                dists.append(dist)
                count_diffs.append(abs(model_counts.get(mk, 0) - exported_mesh_counts.get(fp, 0)))

        max_dist = max(dists) if dists else 1.0
        max_count = max(count_diffs) if count_diffs else 1.0
        if max_dist == 0:
            max_dist = 1.0
        if max_count == 0:
            max_count = 1.0

        beta = float(args.count_weight)  # weight for count penalty
        gamma = float(args.iou_weight)    # weight for IoU penalty (1 - IoU)

        for i, mk in enumerate(models):
            for j, fp in enumerate(files):
                dist = np.linalg.norm(model_centroids[mk] - exported_centroids[fp])
                dist_norm = dist / max_dist
                count_norm = abs(model_counts.get(mk, 0) - exported_mesh_counts.get(fp, 0)) / max_count
                min_a, max_a = model_aabbs[mk]
                min_b, max_b = file_aabbs[fp]
                iou = aabb_iou(min_a, max_a, min_b, max_b)
                iou_penalty = 1.0 - iou
                cost_val = dist_norm + beta * count_norm + gamma * iou_penalty
                cost[i, j] = cost_val

        # Solve assignment
        assignment_rows, assignment_cols = None, None
        hung = try_hungarian(cost)
        if hung is not None:
            r_idx, c_idx = hung
            assignment_rows = list(range(N))
            assignment_cols = [-1] * N
            for r, c in zip(r_idx, c_idx):
                assignment_cols[r] = c
        else:
            r_idx, c_idx = greedy_with_pairwise_improvement(cost)
            assignment_rows = r_idx
            assignment_cols = c_idx

        # Build assignment mapping for actual model keys only
        assignment = {}  # raw_key -> (file_path, cost)
        for i, mk in enumerate(models):
            col = assignment_cols[i] if i < len(assignment_cols) else -1
            if col is None or col < 0 or col >= len(files):
                continue
            assignment[mk] = (files[col], float(cost[i, col]))

        # For any models left unassigned but files remain, try greedy fill (PRESERVED)
        assigned_files = set(v[0] for v in assignment.values())
        remaining_files = [f for f in files if f not in assigned_files]
        remaining_models = [m for m in models if m not in assignment]
        for mk in remaining_models:
            if not remaining_files:
                break
            best_f = None
            best_d = None
            for fp in remaining_files:
                d = np.linalg.norm(model_centroids[mk] - exported_centroids[fp])
                if best_d is None or d < best_d:
                    best_d = d
                    best_f = fp
            if best_f:
                assignment[mk] = (best_f, float(best_d))
                remaining_files.remove(best_f)

        # Now build rename plan
        rename_plan = []
        used_targets = set()
        manifest = {"vox": vox_file, "timestamp": datetime.datetime.now().isoformat(), "renames": []}

        for raw_key, (exp_path, dist) in assignment.items():
            
            # --- CORE LOGICAL FIX: DECOUPLE NAME FROM PARSER'S FLAWED MAP ---
            intended_name = find_canonical_name(raw_key, parser, model_centroids.get(raw_key), model_counts.get(raw_key), max_dist, max_count)
            # --- END FIX ---

            safe = sanitize_filename(intended_name) or raw_key
            target_filename = f"{safe}.obj"
            target_path = os.path.join(exported_dir, target_filename)

            # If same file, no-op
            if os.path.abspath(exp_path) == os.path.abspath(target_path):
                used_targets.add(target_path)
                manifest["renames"].append({"src": os.path.basename(exp_path), "dst": os.path.basename(target_path), "backup": None, "distance": dist})
                continue

            bkp = None
            if os.path.exists(target_path):
                bkp = backup_path(target_path)

            if target_path in used_targets:
                base, ext = os.path.splitext(target_filename)
                idx = 1
                while True:
                    alt = f"{base}_{idx}{ext}"
                    alt_path = os.path.join(exported_dir, alt)
                    if not os.path.exists(alt_path) and alt_path not in used_targets:
                        target_path = alt_path
                        target_filename = alt
                        break
                    idx += 1

            used_targets.add(target_path)
            rename_plan.append((exp_path, target_path, bkp, raw_key, intended_name, dist))
            manifest["renames"].append({"src": os.path.basename(exp_path), "dst": os.path.basename(target_path), "backup": os.path.basename(bkp) if bkp else None, "distance": dist})

        # Write report and manifest (dry-run content)
        with open(report_path, "w", encoding="utf-8") as rep:
            rep.write(f"RenamingReport (dry-run={not args.commit}) for: {vox_file}\n\n")
            rep.write("Assignments (raw_key -> exported_file, distance):\n")
            for raw_key, (exp_path, dist) in assignment.items():
                rep.write(f"  {raw_key} -> {os.path.basename(exp_path)}  dist={dist}\n")
            rep.write("\nPlanned renames:\n")
            for cur, tgt, bkp, raw_key, intended, dist in rename_plan:
                rep.write(f"  {os.path.basename(cur)} -> {os.path.basename(tgt)}  (raw {raw_key} => '{intended}', dist={dist})")
                if bkp:
                    rep.write(f"  [will backup existing {os.path.basename(tgt)} -> {os.path.basename(bkp)}]")
                rep.write("\n")
            rep.write("\nManifest file written next to this report for automatic undo.\n")

        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        print(f"  Report written: {report_path}")
        print(f"  Manifest written: {manifest_path}")
        if not rename_plan:
            print("  Nothing to rename.")
            
        # Perform renames when --commit provided (two-phase safe rename)
        if args.commit:
            temp_map = []
            try:
                # Phase 1: move sources to temps
                for cur, tgt, bkp, raw_key, intended, dist in rename_plan:
                    cur_abs = os.path.abspath(cur)
                    tgt_abs = os.path.abspath(tgt)
                    if not os.path.exists(cur_abs):
                        with open(report_path, "a", encoding="utf-8") as rep:
                            rep.write(f"\nERROR: source not found (skipping): {cur_abs}\n")
                        print(f"  ERROR: source not found (skipping): {cur_abs}")
                        continue
                    if cur_abs == tgt_abs:
                        temp_map.append((None, tgt_abs, None, cur))
                        continue
                    tmp_name = f".tmp_rename_{uuid.uuid4().hex}.obj"
                    tmp_path = os.path.join(exported_dir, tmp_name)
                    shutil.move(cur_abs, tmp_path)
                    temp_map.append((tmp_path, tgt_abs, bkp, cur))
                # Phase 2: apply backups and move temps to final targets
                for tmp_path, final_tgt, backup_for, original_src in temp_map:
                    if tmp_path is None:
                        continue
                    if os.path.exists(final_tgt):
                        bkp_path = backup_path(final_tgt)
                        shutil.move(final_tgt, bkp_path)
                        src_name = os.path.basename(original_src)
                        for m in manifest["renames"]:
                            if m["src"] == src_name:
                                m["backup"] = os.path.basename(bkp_path)
                                break
                    shutil.move(tmp_path, final_tgt)
                # Save manifest after actual moves
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    json.dump(manifest, mf, indent=2)
                print(f"  Completed renames (backups created for overwritten files). See {report_path}")
            except Exception as ex:
                with open(report_path, "a", encoding="utf-8") as rep:
                    rep.write(f"\nERROR during rename operation: {ex}\n")
                print(f"  ERROR during rename operation: {ex}")
        else:
            print("  Dry-run only. Re-run with --commit to apply renames (backups will be created for overwritten files).")
            
    except Exception as ex:
        print(f"\n[FATAL ERROR]: An unhandled exception occurred during processing of {vox_file}:")
        print(ex)
        print(traceback.format_exc())
        raise
