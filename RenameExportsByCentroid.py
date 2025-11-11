# -*- coding: utf-8 -*-
"""
RenameExportsByCentroid.py
Safe, no-scaling renamer:
- Reads parser intent and exported OBJ centroids/AABBs only (no geometry modification).
- Preserves descriptive names already assigned to non-suspect parts (1:1 protection).
- Transfers descriptive names from tiny helper parts (donors) to large winners when appropriate.
- Dry-run by default; --commit performs atomic file renames and creates backups.
"""
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

# Safety guard — prevent accidental geometry mutation from this script.
_mutating_methods = [
    "apply_scale", "apply_translation", "apply_transform",
    "export", "export_obj", "export_wkb", "export_wkt",
    "remove_degenerate_faces", "remove_duplicate_faces",
    "update_faces", "update_vertices",
]
def _raise_mutation(*args, **kwargs):
    raise RuntimeError("RenameExportsByCentroid.py is prohibited from modifying mesh geometry or exporting meshes.")
for _m in _mutating_methods:
    if hasattr(trimesh.Trimesh, _m):
        try:
            setattr(trimesh.Trimesh, _m, _raise_mutation)
        except Exception:
            pass

def sanitize_filename(name):
    return ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in (name or "")).strip()

BASE_DIR = os.path.dirname(__file__)
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

parser_arg = argparse.ArgumentParser(description="Rename exported .obj files to match parser intent by safe helper->large transfers.")
parser_arg.add_argument("--commit", action="store_true", help="Perform renames (default is dry-run). Backups are created for overwritten files.")
parser_arg.add_argument("--vox", help="Optional: limit to a single .vox basename (no extension).")
parser_arg.add_argument("--suspect-threshold", type=int, default=10, help="Max voxel count for a part to be considered a suspect helper (default 10).")
parser_arg.add_argument("--size-ratio-threshold", type=float, default=4.0, help="Required ratio between winning part and suspect donor (default 4.0).")
parser_arg.add_argument("--centroid-threshold", type=float, default=10.0, help="Max centroid distance for non-essential transfers (default 10.0).")
parser_arg.add_argument("--name-trust-threshold", type=int, default=100, help="If winning part is larger than this, it may override distance checks (default 100).")
parser_arg.add_argument("--allow-transfer", nargs="*", default=[], help="Explicit donor names to allow transfer even if protected (e.g. --allow-transfer Head)")
parser_arg.add_argument("--debug", action="store_true", help="Emit debug claim info to console and include in report")
parser_arg.add_argument("--active", choices=["0","1","true","false"], help="Enable/disable the renamer (overrides env RENAME_ACTIVE)")
args = parser_arg.parse_args()

# Determine active state: CLI --active > ENV RENAME_ACTIVE > default True
env_active = os.environ.get("RENAME_ACTIVE")
cli_active = args.active
active_value = cli_active if cli_active is not None else (env_active if env_active is not None else "1")
RENAMER_ACTIVE = not str(active_value).lower() in ("0", "false", "no", "n")
if not RENAMER_ACTIVE:
    dryrun_log = os.path.join(REPORTS_DIR, "RenameDryrun_Character.log")
    with open(dryrun_log, "w", encoding="utf-8") as f:
        f.write("RenameExportsByCentroid.py: renamer disabled by environment/CLI (RENAME_ACTIVE=0).\nDry-run skipped.\n")
    print("[SKIP] RenameExportsByCentroid.py disabled by RENAME_ACTIVE=0; wrote RenameDryrun_Character.log")
    raise SystemExit(0)

HIGH_PRIORITY_NAMES = set(["head", "face"])

def backup_path(path):
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    base = os.path.basename(path)
    return os.path.join(os.path.dirname(path), f"{base}.bak.{ts}")

def aabb_iou(min_a, max_a, min_b, max_b):
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

def looks_descriptive(name):
    if not name:
        return False
    s = str(name).strip()
    if len(s) <= 2:
        return False
    if re.fullmatch(r'^\d+$', s):
        return False
    if len(s) <= 3 and any(c.isdigit() for c in s):
        return False
    return True

def transfer_suspect_names(parser, model_centroids, model_counts):
    allow_list = set([str(x) for x in (args.allow_transfer or [])])

    protected_names = set()
    for rk, nm in parser.layer_name_map.items():
        if nm and not str(nm).startswith("Part_"):
            protected_names.add(str(nm))

    for rk, nm in parser.raw_part_name_map.items():
        if nm and looks_descriptive(nm):
            cnt = model_counts.get(rk, 0)
            if cnt > args.suspect_threshold:
                protected_names.add(str(nm))

    suspect_donors = {}
    trusted_winners = {}
    for rk, nm in parser.raw_part_name_map.items():
        if not looks_descriptive(nm):
            continue
        cnt = model_counts.get(rk, 0)
        name_str = str(nm)
        if cnt <= args.suspect_threshold:
            if name_str in protected_names and name_str not in allow_list:
                nl = name_str.lower()
                if nl in HIGH_PRIORITY_NAMES:
                    donor_cnt = cnt
                    big_candidate_exists = any(
                        (mc >= args.name_trust_threshold and mc >= math.ceil(args.size_ratio_threshold * max(1, donor_cnt)))
                        for mc in model_counts.values()
                    )
                    if not big_candidate_exists:
                        if args.debug:
                            print(f"[SKIP_PROTECTED_DONOR] {rk} name='{nm}' protected and no big candidate; skipping donor.")
                        continue
                else:
                    if args.debug:
                        print(f"[SKIP_PROTECTED_DONOR] {rk} name='{nm}' is protected; not treated as donor.")
                    continue
            suspect_donors[name_str] = rk
        else:
            trusted_winners[rk] = nm

    if not suspect_donors or not trusted_winners:
        return {}

    def donor_sort_key(item):
        name, key = item
        nl = name.lower()
        priority = 0 if any(tok in nl for tok in HIGH_PRIORITY_NAMES) else 1
        cnt = model_counts.get(key, 0)
        return (priority, cnt)
    donor_items = sorted(list(suspect_donors.items()), key=donor_sort_key)
    sorted_winners = sorted(trusted_winners.keys(), key=lambda k: model_counts.get(k, 0), reverse=True)

    transfers = {}
    claimed = set()

    for donor_name, donor_key in donor_items:
        if donor_name in claimed:
            continue
        donor_count = model_counts.get(donor_key, 0)
        donor_centroid = model_centroids.get(donor_key)
        if donor_centroid is None:
            continue

        for winner_key in sorted_winners:
            if winner_key in transfers:
                continue

            winner_count = model_counts.get(winner_key, 0)
            winner_parser_name = parser.layer_name_map.get(winner_key)
            if winner_parser_name and not str(winner_parser_name).startswith("Part_"):
                nl_donor = donor_name.lower()
                donor_allowed = (donor_name in allow_list) or (nl_donor in HIGH_PRIORITY_NAMES)
                if not donor_allowed or winner_count < args.name_trust_threshold:
                    if args.debug:
                        print(f"[SKIP_WINNER_ALREADY_NAMED] winner {winner_key} already has parser name '{winner_parser_name}'; donor_allowed={donor_allowed} winner_count={winner_count}")
                    continue

            if winner_count < math.ceil(args.size_ratio_threshold * max(1, donor_count)):
                continue

            winner_centroid = model_centroids.get(winner_key)
            if winner_centroid is None:
                continue

            dist = float(np.linalg.norm(winner_centroid - donor_centroid))
            max_dist_check = args.centroid_threshold if winner_count < args.name_trust_threshold else float("inf")

            if dist > max_dist_check:
                if args.debug:
                    print(f"[DEBUG] donor {donor_key} '{donor_name}' skip {winner_key} dist {dist:.2f} > {max_dist_check:.2f}")
                continue

            transfers[winner_key] = donor_name
            claimed.add(donor_name)
            if args.debug:
                print(f"[TRANSFER] donor {donor_key} (count={donor_count}) -> winner {winner_key} (count={winner_count}) name='{donor_name}' dist={dist:.2f}")
            break

    return transfers

# Main driver
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
        voxels_by_layer = parser.voxels_by_layer
        layer_name_map = parser.layer_name_map
        raw_part_name_map = parser.raw_part_name_map
        model_keys = [k for k in voxels_by_layer.keys() if len(voxels_by_layer[k]) > 0]

        # per-model stats
        model_centroids = {}
        model_counts = {}
        model_aabbs = {}
        for raw_key in model_keys:
            vox = voxels_by_layer[raw_key]
            pts = np.array([[v.x, v.y, v.z] for v in vox], dtype=float)
            model_centroids[raw_key] = pts.mean(axis=0)
            model_counts[raw_key] = len(vox)
            model_aabbs[raw_key] = (pts.min(axis=0), pts.max(axis=0))

        # compute transfers
        transfers = transfer_suspect_names(parser, model_centroids, model_counts)
        transfers_debug = [f"[TRANSFER_ASSIGNED] {k} => '{v}'" for k, v in transfers.items()]

        # exported OBJ files
        exported_dir = os.path.join(EXPORT_ROOT, vox_base)
        exported_paths = sorted(glob.glob(os.path.join(exported_dir, "*.obj")))
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
            continue

        exported_centroids = {}
        exported_mesh_counts = {}
        file_aabbs = {}
        for p in exported_paths:
            try:
                m = trimesh.load(p, force='mesh')
                c = np.array(m.centroid if hasattr(m, "centroid") else np.array(m.bounds).mean(axis=0), dtype=float)
                exported_centroids[p] = c
                file_aabbs[p] = (m.bounds[0].astype(float), m.bounds[1].astype(float))
            except Exception:
                exported_centroids[p] = np.array([math.inf, math.inf, math.inf], dtype=float)
                file_aabbs[p] = (np.array([math.inf, math.inf, math.inf], dtype=float), np.array([-math.inf, -math.inf, -math.inf], dtype=float))

        # optional: extract counts from ExportLog if present
        exportlog_counts = {}
        exp_log = os.path.join(exported_dir, "ExportLog.txt")
        if os.path.exists(exp_log):
            try:
                with open(exp_log, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        m = re.search(r'Part_(\d+).*?(\d+)\s+voxels', line)
                        if m:
                            fname = f"Part_{m.group(1)}.obj"
                            exportlog_counts[fname] = int(m.group(2))
            except Exception:
                pass

        for p in exported_paths:
            fname = os.path.basename(p)
            exported_mesh_counts[p] = exportlog_counts.get(fname, 0)

        # cost matrix
        models = model_keys
        files = exported_paths
        R = len(models)
        C = len(files)
        N = max(R, C)
        cost = np.full((N, N), fill_value=1e6, dtype=float)

        dists = []
        count_diffs = []
        for i, mk in enumerate(models):
            for j, fp in enumerate(files):
                dist = np.linalg.norm(model_centroids[mk] - exported_centroids[fp])
                dists.append(dist)
                count_diffs.append(abs(model_counts.get(mk, 0) - exported_mesh_counts.get(fp, 0)))

        max_dist = max(dists) if dists else 1.0
        max_count = max(count_diffs) if count_diffs else 1.0
        if max_dist == 0: max_dist = 1.0
        if max_count == 0: max_count = 1.0

        beta = 0.5
        gamma = 1.0
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

        # assignment (Hungarian or greedy fallback)
        try:
            from scipy.optimize import linear_sum_assignment
            r_idx, c_idx = linear_sum_assignment(cost)
            assignment_cols = [-1] * N
            for r, c in zip(r_idx, c_idx):
                assignment_cols[r] = c
        except Exception:
            def greedy(cost):
                n_rows, n_cols = cost.shape
                assigned_cols = set()
                assignment = [-1] * n_rows
                for r in range(n_rows):
                    order = sorted(range(n_cols), key=lambda c: cost[r, c])
                    for c in order:
                        if c not in assigned_cols:
                            assignment[r] = c
                            assigned_cols.add(c)
                            break
                return assignment
            assignment_cols = greedy(cost)

        assignment = {}
        for i, mk in enumerate(models):
            col = assignment_cols[i] if i < len(assignment_cols) else -1
            if col is None or col < 0 or col >= len(files):
                continue
            assignment[mk] = (files[col], float(cost[i, col]))

        # Build rename plan (transfers take precedence)
        rename_plan = []
        used_targets = set()
        manifest = {"vox": vox_file, "timestamp": datetime.datetime.now().isoformat(), "renames": []}

        for raw_key, (exp_path, dist) in assignment.items():
            # transfers override parser names
            intended_name = transfers.get(raw_key)
            if not intended_name:
                pname = layer_name_map.get(raw_key)
                if pname and not str(pname).startswith("Part_"):
                    intended_name = pname
            if not intended_name:
                intended_name = raw_key

            safe = sanitize_filename(intended_name) or raw_key
            target_filename = f"{safe}.obj"
            target_path = os.path.join(exported_dir, target_filename)

            if os.path.abspath(exp_path) == os.path.abspath(target_path):
                used_targets.add(target_path)
                manifest["renames"].append({"src": os.path.basename(exp_path), "dst": os.path.basename(target_path), "backup": None, "distance": dist})
                continue

            bkp = None
            if os.path.exists(target_path):
                bkp = backup_path(target_path)

            if target_path in used_targets:
                base, ext = os.path.splitext(target_filename)
                idxdup = 1
                while True:
                    alt = f"{base}_{idxdup}{ext}"
                    alt_path = os.path.join(exported_dir, alt)
                    if not os.path.exists(alt_path) and alt_path not in used_targets:
                        target_path = alt_path
                        target_filename = alt
                        break
                    idxdup += 1

            used_targets.add(target_path)
            rename_plan.append((exp_path, target_path, bkp, raw_key, intended_name, dist))
            manifest["renames"].append({"src": os.path.basename(exp_path), "dst": os.path.basename(target_path), "backup": os.path.basename(bkp) if bkp else None, "distance": dist})

        # Write report & manifest
        with open(report_path, "w", encoding="utf-8") as rep:
            rep.write(f"RenamingReport (dry-run={not args.commit}) for: {vox_file}\n\n")
            rep.write("Transfers (surgical):\n")
            if transfers_debug:
                for ln in transfers_debug:
                    rep.write("  " + ln + "\n")
            else:
                rep.write("  (none)\n")
            rep.write("\nAssignments (raw_key -> exported_file, distance):\n")
            for raw_key, (exp_path, dist) in assignment.items():
                rep.write(f"  {raw_key} -> {os.path.basename(exp_path)}  dist={dist}\n")
            rep.write("\nPlanned renames:\n")
            for cur, tgt, bkp, raw_key, intended, dist in rename_plan:
                rep.write(f"  {os.path.basename(cur)} -> {os.path.basename(tgt)}  (raw {raw_key} => '{intended}', dist={dist})")
                if bkp:
                    rep.write(f"  [will backup existing {os.path.basename(tgt)} -> {os.path.basename(bkp)}]")
                rep.write("\n")

        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        if args.debug and transfers_debug:
            print("\n[CLAIM DEBUG]")
            for ln in transfers_debug:
                print(" ", ln)

        print(f"  Report written: {report_path}")
        print(f"  Manifest written: {manifest_path}")
        if not rename_plan:
            print("  Nothing to rename.")
        else:
            if args.commit:
                temp_map = []
                try:
                    for cur, tgt, bkp, raw_key, intended, dist in rename_plan:
                        cur_abs = os.path.abspath(cur)
                        tgt_abs = os.path.abspath(tgt)
                        if not os.path.exists(cur_abs):
                            print(f"  ERROR: source not found (skipping): {cur_abs}")
                            continue
                        if cur_abs == tgt_abs:
                            temp_map.append((None, tgt_abs, None, cur))
                            continue
                        tmp_name = f".tmp_rename_{uuid.uuid4().hex}.obj"
                        tmp_path = os.path.join(exported_dir, tmp_name)
                        shutil.move(cur_abs, tmp_path)
                        temp_map.append((tmp_path, tgt_abs, bkp, cur))
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
                    with open(manifest_path, "w", encoding="utf-8") as mf:
                        json.dump(manifest, mf, indent=2)
                    print(f"  Completed renames (backups created for overwritten files). See {report_path}")
                except Exception as ex:
                    print(f"  ERROR during rename operation: {ex}")
            else:
                print("  Dry-run only. Re-run with --commit to apply renames (backups will be created for overwritten files).")

    except Exception as ex:
        print(f"\n[FATAL ERROR]: An unhandled exception occurred during processing of {vox_file}:")
        print(ex)
        print(traceback.format_exc())
        raise