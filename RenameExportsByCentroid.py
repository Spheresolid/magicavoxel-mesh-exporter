import os
import glob
import shutil
import math
import argparse
import datetime
import json
import numpy as np
import trimesh
from Vox200Parser import Vox200Parser

def sanitize_filename(name):
    return ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in (name or "")).strip()

BASE_DIR = os.path.dirname(__file__)
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

parser_arg = argparse.ArgumentParser(description="Rename exported .obj files to match parser intent by centroid.")
parser_arg.add_argument("--commit", action="store_true", help="Perform renames (default is dry-run). Backups are created for overwritten files.")
parser_arg.add_argument("--vox", help="Optional: limit to a single .vox basename (no extension).")
args = parser_arg.parse_args()

def backup_path(path):
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    base = os.path.basename(path)
    return os.path.join(os.path.dirname(path), f"{base}.bak.{ts}")

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
    vox_path = os.path.join(BASE_DIR, vox_file)
    vox_base = os.path.splitext(vox_file)[0]
    print(f"Processing: {vox_file}")

    parser = Vox200Parser(vox_path).parse()
    voxels_by_layer = parser.voxels_by_layer         # dict 'Part_X' -> list of Voxels
    name_map = parser.layer_name_map                 # 'Part_X' -> intended name

    # Build model centroids for non-empty parts
    model_centroids = {}
    for raw_key, voxels in voxels_by_layer.items():
        if not voxels:
            continue
        pts = np.array([[v.x, v.y, v.z] for v in voxels], dtype=float)
        model_centroids[raw_key] = pts.mean(axis=0)

    exported_dir = os.path.join(EXPORT_ROOT, vox_base)
    exported_paths = sorted(glob.glob(os.path.join(exported_dir, "*.obj")))
    if not exported_paths:
        print(f"  No exported .obj files found in {exported_dir}, skipping.")
        continue

    # Compute exported mesh centroids
    exported_centroids = {}
    for p in exported_paths:
        try:
            m = trimesh.load(p, force='mesh')
            c = np.array(m.centroid if hasattr(m, "centroid") else np.array(m.bounds).mean(axis=0), dtype=float)
            exported_centroids[p] = c
        except Exception:
            exported_centroids[p] = np.array([math.inf, math.inf, math.inf], dtype=float)

    # Greedy nearest-centroid assignment
    remaining_files = set(exported_paths)
    assignment = {}
    for raw_key, mcent in model_centroids.items():
        best_file, best_dist = None, None
        for p in list(remaining_files):
            d = np.linalg.norm(exported_centroids[p] - mcent)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_file = p
        if best_file:
            assignment[raw_key] = (best_file, float(best_dist))
            remaining_files.remove(best_file)

    # Any leftover models/files handle conservatively (assign remaining files arbitrarily)
    unassigned_models = [k for k in voxels_by_layer.keys() if k not in assignment and voxels_by_layer.get(k)]
    for raw_key in unassigned_models:
        if remaining_files:
            p = remaining_files.pop()
            assignment[raw_key] = (p, float('nan'))

    # Build rename plan and manifest entries
    rename_plan = []
    used_targets = set()
    manifest = {"vox": vox_file, "timestamp": datetime.datetime.now().isoformat(), "renames": []}

    for raw_key, (exp_path, dist) in assignment.items():
        intended_name = name_map.get(raw_key) or raw_key
        safe = sanitize_filename(intended_name) or raw_key
        target_filename = f"{safe}.obj"
        target_path = os.path.join(exported_dir, target_filename)

        # If an export currently already has the target name and it's the same file, no-op.
        if os.path.abspath(exp_path) == os.path.abspath(target_path):
            used_targets.add(target_path)
            manifest["renames"].append({"src": os.path.basename(exp_path), "dst": os.path.basename(target_path), "backup": None, "distance": dist})
            continue

        # If target exists and is a different file, plan backup
        bkp = None
        if os.path.exists(target_path):
            bkp = backup_path(target_path)

        # If another raw_key already plans to target this path, append numeric suffix
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

    # Write detailed report and manifest
    report_path = os.path.join(REPORTS_DIR, f"RenamingReport_{vox_base}.txt")
    manifest_path = os.path.join(REPORTS_DIR, f"RenamingManifest_{vox_base}.json")
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
        continue

    # Perform renames when --commit provided
    if args.commit:
        for cur, tgt, bkp, raw_key, intended, dist in rename_plan:
            try:
                # Backup existing target if any
                if bkp and os.path.exists(tgt):
                    shutil.move(tgt, bkp)
                # Move current file to target
                if os.path.exists(cur):
                    shutil.move(cur, tgt)
                else:
                    with open(report_path, "a", encoding="utf-8") as rep:
                        rep.write(f"\nERROR: source not found: {cur}\n")
                    print(f"  ERROR: source not found: {cur}")
            except Exception as ex:
                with open(report_path, "a", encoding="utf-8") as rep:
                    rep.write(f"\nERROR renaming {cur} -> {tgt}: {ex}\n")
                print(f"  ERROR renaming {cur} -> {tgt}: {ex}")
        print(f"  Completed renames (backups created for overwritten files). See {report_path}")
    else:
        print("  Dry-run only. Re-run with --commit to apply renames (backups will be created for overwritten files).")