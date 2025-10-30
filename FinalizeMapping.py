import os
import glob
import csv
import numpy as np
import trimesh
from Vox200Parser import Vox200Parser

def sanitize_filename(name):
    return ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in (name or "")).strip()

BASE_DIR = os.path.dirname(__file__)
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]
if not vox_files:
    print("No .vox files found.")
    raise SystemExit(0)

for vf in vox_files:
    vox_path = os.path.join(BASE_DIR, vf)
    vox_base = os.path.splitext(vf)[0]
    print(f"Finalizing mapping for: {vf}")

    parser = Vox200Parser(vox_path).parse()
    voxels_by_layer = parser.voxels_by_layer          # Part_X -> [Voxel]
    name_map = parser.layer_name_map                  # Part_X -> intended name

    # compute part centroids
    part_centroids = {}
    part_counts = {}
    for raw_key, voxels in voxels_by_layer.items():
        if not voxels:
            continue
        pts = np.array([[v.x, v.y, v.z] for v in voxels], dtype=float)
        part_centroids[raw_key] = pts.mean(axis=0)
        part_counts[raw_key] = len(voxels)

    exported_dir = os.path.join(EXPORT_ROOT, vox_base)
    exported_paths = sorted(glob.glob(os.path.join(exported_dir, "*.obj")))
    if not exported_paths:
        print(f"  No exported .obj files found for {vox_base}")
        continue

    # compute exported mesh centroids
    exported_centroids = {}
    for p in exported_paths:
        try:
            m = trimesh.load(p, force='mesh')
            c = np.array(m.centroid if hasattr(m, "centroid") else np.array(m.bounds).mean(axis=0), dtype=float)
            exported_centroids[p] = c
        except Exception:
            exported_centroids[p] = np.array([np.nan, np.nan, np.nan], dtype=float)

    # match exported files to nearest part centroid (greedy)
    remaining_parts = set(part_centroids.keys())
    remaining_files = set(exported_centroids.keys())
    assignments = []  # tuples (export_filename, raw_part, intended_name, vox_count, distance)

    # greedy: for each exported file find nearest part not yet taken
    for p in sorted(exported_centroids.keys()):
        best_part = None
        best_dist = None
        ep = exported_centroids[p]
        for part in remaining_parts:
            dist = np.linalg.norm(ep - part_centroids[part])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_part = part
        if best_part is not None:
            assignments.append((os.path.basename(p), best_part, name_map.get(best_part, best_part), part_counts.get(best_part, 0), float(best_dist)))
            remaining_parts.remove(best_part)
            remaining_files.discard(p)
        else:
            assignments.append((os.path.basename(p), None, None, 0, float('nan')))

    # any leftover parts (no files) add to report
    for part in sorted(remaining_parts):
        assignments.append(("<no-file>", part, name_map.get(part, part), part_counts.get(part, 0), float('nan')))

    # write text and CSV reports
    txt_path = os.path.join(REPORTS_DIR, f"FinalMapping_{vox_base}.txt")
    csv_path = os.path.join(REPORTS_DIR, f"FinalMapping_{vox_base}.csv")
    with open(txt_path, "w", encoding="utf-8") as t:
        t.write(f"Final Mapping for: {vf}\n\n")
        t.write("ExportFile, RawPart, IntendedName, VoxelCount, CentroidDistance\n")
        for row in assignments:
            t.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]:.6f}\n")
    with open(csv_path, "w", newline='', encoding="utf-8") as c:
        writer = csv.writer(c)
        writer.writerow(["ExportFile","RawPart","IntendedName","VoxelCount","CentroidDistance"])
        for row in assignments:
            writer.writerow(row)

    print(f"  Wrote: {txt_path} and {csv_path}")