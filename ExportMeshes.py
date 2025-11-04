# ExportMeshes.py
import os
import sys
import math
import traceback
import csv
import glob

import numpy as np
import trimesh

from Vox200Parser import Vox200Parser

# ASCII-safe sanitizer for filenames
def sanitize_filename(name):
    if not name:
        return ""
    s = str(name)
    # keep basic ASCII letters/digits and a few symbols; replace others with underscore
    safe = "".join(c if (c.isalnum() or c in (' ', '.', '_', '-')) else '_' for c in s)
    while '__' in safe:
        safe = safe.replace('__', '_')
    safe = safe.strip().replace(' ', '_')
    return safe or ""

BASE_DIR = os.path.dirname(__file__)
os.chdir(BASE_DIR)

EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
os.makedirs(EXPORT_ROOT, exist_ok=True)

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def compute_centroid(voxels):
    pts = np.array([[v.x, v.y, v.z] for v in voxels], dtype=float)
    if pts.size == 0:
        return np.array([math.nan, math.nan, math.nan], dtype=float)
    return pts.mean(axis=0)

def load_reference_mappings(vox_name):
    """
    Load reference mappings from FinalMapping_<vox>.csv (preferred) or DebugNameMap_<vox>.txt.
    Returns list of entries: { 'raw': 'Part_26', 'name': 'Head', 'count': 474 }
    """
    refs = []
    csv_path = os.path.join(REPORTS_DIR, f"FinalMapping_{vox_name}.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    raw = (r.get("RawPart") or r.get(" RawPart") or "").strip()
                    nm = (r.get("IntendedName") or "").strip()
                    cnt = r.get("VoxelCount")
                    try:
                        cntv = int(cnt) if cnt not in (None, "") else None
                    except Exception:
                        cntv = None
                    if raw and nm:
                        refs.append({"raw": raw, "name": nm, "count": cntv})
            return refs
        except Exception:
            pass

    # Fallback: parse DebugNameMap_<vox>.txt for "Final assignment (Part_X (N voxels) => Name)" lines.
    dbg_path = os.path.join(REPORTS_DIR, f"DebugNameMap_{vox_name}.txt")
    if os.path.exists(dbg_path):
        try:
            with open(dbg_path, "r", encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    # expect lines like: Part_26 (474 voxels) => Right_UpperLeg
                    if line.startswith("Part_") and "=>" in line:
                        try:
                            left, right = line.split("=>", 1)
                            left = left.strip()
                            right = right.strip()
                            # extract Part_N and count
                            if "(" in left and "voxels" in left:
                                part = left.split("(", 1)[0].strip()
                                cnt_part = left.split("(", 1)[1].split("voxels",1)[0].strip()
                                try:
                                    cntv = int(cnt_part)
                                except Exception:
                                    cntv = None
                            else:
                                part = left
                                cntv = None
                            refs.append({"raw": part, "name": right, "count": cntv})
                        except Exception:
                            continue
            return refs
        except Exception:
            pass

    return refs

def find_best_ref_by_count(refs, count, tol_abs=3, tol_rel=0.02):
    """
    Find a single best reference entry matching `count`.
    Returns entry or None. Uses absolute and relative tolerance.
    """
    if count is None:
        return None
    matches = []
    for e in refs:
        ec = e.get("count")
        if ec is None:
            continue
        # compute difference
        diff = abs(ec - count)
        rel = diff / max(1.0, ec)
        if diff <= tol_abs or rel <= tol_rel:
            matches.append((diff, rel, e))
    if not matches:
        return None
    # choose smallest absolute diff then smallest relative
    matches.sort(key=lambda x: (x[0], x[1]))
    # if top two entries have same diff and same rel, ambiguous -> skip
    if len(matches) > 1 and matches[0][0] == matches[1][0] and matches[0][1] == matches[1][1]:
        return None
    return matches[0][2]

# Main export
vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]
if not vox_files:
    print("No .vox files found.")
    sys.exit(0)

for vox_file in vox_files:
    vox_path = os.path.join(BASE_DIR, vox_file)
    vox_name = os.path.splitext(vox_file)[0]
    sub_export_dir = os.path.join(EXPORT_ROOT, vox_name)
    os.makedirs(sub_export_dir, exist_ok=True)

    # CLEANUP: remove existing .obj files to avoid accumulation across runs
    try:
        for old in glob.glob(os.path.join(sub_export_dir, "*.obj")):
            try:
                os.remove(old)
            except Exception:
                pass
    except Exception:
        pass

    try:
        parser = Vox200Parser(vox_path).parse()
    except Exception as ex:
        print(f"Failed to parse {vox_file}: {ex}")
        continue

    voxels_by_layer = parser.voxels_by_layer    # dict Part_X -> [Voxel]
    name_map = parser.layer_name_map            # dict Part_X -> intended name

    # load reference mappings to assist overrides
    refs = load_reference_mappings(vox_name)

    # Precompute stats and build ordered list of parts by numeric index
    parts = []
    for k in voxels_by_layer.keys():
        try:
            idx = int(k.split("_", 1)[1])
        except Exception:
            idx = None
        parts.append((idx, k))
    parts.sort(key=lambda x: (x[0] if x[0] is not None else 1e9, x[1]))
    ordered_parts = [p[1] for p in parts]

    stats = {}
    for raw_key in ordered_parts:
        voxels = voxels_by_layer.get(raw_key, [])
        cnt = len(voxels)
        cent = compute_centroid(voxels) if cnt > 0 else None
        stats[raw_key] = {"count": cnt, "centroid": cent}

    # Prepare reports
    export_log_path = os.path.join(sub_export_dir, "ExportLog.txt")
    override_report_path = os.path.join(REPORTS_DIR, f"VoxelOverrideReport_{vox_name}.txt")
    layer_map_path = os.path.join(REPORTS_DIR, f"LayerMapping_{vox_name}.txt")

    # Write LayerMapping report (ASCII-safe)
    try:
        with open(layer_map_path, "w", encoding="utf-8", errors="replace") as lm:
            lm.write(f"LayerMapping for: {vox_file}\n\n")
            lm.write(f"{'Part':<12} {'VoxelCount':>10} {'ParserAssigned':<30}\n")
            lm.write(f"{'-'*12} {'-'*10} {'-'*30}\n")
            for raw_key in ordered_parts:
                assigned = name_map.get(raw_key, raw_key)
                cnt = stats[raw_key]["count"]
                lm.write(f"{raw_key:<12} {cnt:10d} {str(assigned):<30}\n")
    except Exception as ex:
        print(f"Failed to write layer mapping report: {ex}")

    # Start export log and override report
    try:
        with open(export_log_path, "w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"Export Status & Layer Name Mapping for {vox_file}\n\n")
    except Exception as ex:
        print(f"Failed to create export log: {ex}")
        continue

    try:
        orp = open(override_report_path, "w", encoding="utf-8", errors="replace")
        orp.write(f"Voxel Override Report for: {vox_file}\n\n")
        orp.write("Part, VoxelCount, Centroid, ParserName, FinalName, NameSource, Reason\n")
    except Exception as ex:
        orp = None
        print(f"Failed to create override report: {ex}")

    used_filenames = set()

    # For each part, decide final name (possibly overridden by voxel metrics)
    for raw_key in ordered_parts:
        voxels = voxels_by_layer.get(raw_key, [])
        voxel_count = len(voxels)
        if voxel_count == 0:
            with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"Skipping empty model: {raw_key}\n")
            if orp:
                orp.write(f"{raw_key},0,,{name_map.get(raw_key)},,skipped,empty\n")
            continue

        parser_assigned = name_map.get(raw_key) or ""
        final_name = parser_assigned
        name_source = "layer_name_map"
        reason = ""

        # If we have reference mappings, try to match by exact count first, then fuzzy
        if refs:
            best_ref = find_best_ref_by_count(refs, voxel_count, tol_abs=3, tol_rel=0.02)
            if best_ref:
                # If the best_ref raw equals current raw_key, keep parser assigned (no-op)
                if best_ref["raw"] != raw_key:
                    # Use the referenced name if it provides a clearer mapping
                    final_name = best_ref["name"]
                    name_source = "finalmapping_override"
                    reason = f"count_match (ref {best_ref['raw']} count={best_ref.get('count')})"
                else:
                    # exact ref points to same raw_key: honor parser (or ref) name
                    final_name = best_ref["name"] or parser_assigned
                    name_source = "finalmapping_confirm"
                    reason = "ref_confirm_same_raw"
            else:
                # No reliable count match. If parser_assigned looks like a helper/placeholder, try fuzzy by any ref with same or similar count.
                # Also if parser_assigned is missing or equals raw_key, attempt to find unique ref with same count.
                if (not parser_assigned) or (parser_assigned == raw_key) or parser_assigned.startswith("_"):
                    fuzzy = find_best_ref_by_count(refs, voxel_count, tol_abs=max(3, int(0.01*voxel_count)), tol_rel=0.05)
                    if fuzzy and fuzzy["raw"] != raw_key:
                        final_name = fuzzy["name"]
                        name_source = "finalmapping_fuzzy"
                        reason = f"fuzzy_count_match (ref {fuzzy['raw']} count={fuzzy.get('count')})"

        # Fallback: ensure final_name non-empty
        if not final_name:
            final_name = parser_assigned or raw_key
            name_source = name_source or "fallback"

        safe_name = sanitize_filename(final_name) or raw_key
        base = safe_name
        suffix = 1
        target_path = os.path.join(sub_export_dir, f"{safe_name}.obj")
        while target_path in used_filenames or os.path.exists(target_path):
            safe_name = f"{base}_{suffix}"
            target_path = os.path.join(sub_export_dir, f"{safe_name}.obj")
            suffix += 1
        used_filenames.add(target_path)

        # Export geometry
        try:
            positions = np.array([[v.x, v.y, v.z] for v in voxels])
            cubes = []
            for pos in positions:
                cube = trimesh.creation.box(extents=(1, 1, 1))
                cube.apply_translation(pos)
                cubes.append(cube)

            if cubes:
                combined = trimesh.util.concatenate(cubes)
                combined.export(target_path)

            with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"Exported {raw_key} ({voxel_count} voxels) as {os.path.basename(target_path)} (source={name_source})\n")

            if orp:
                cent = stats[raw_key]["centroid"] if raw_key in stats else None
                cent_str = ",".join([f"{x:.3f}" for x in cent]) if cent is not None else ""
                orp.write(f"{raw_key},{voxel_count},{cent_str},{parser_assigned},{final_name},{name_source},{reason}\n")
        except Exception as ex:
            with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"ERROR exporting {raw_key}: {ex}\n")
                log_file.write(traceback.format_exc() + "\n")
            if orp:
                orp.write(f"{raw_key},{voxel_count},,ERROR_EXPORT,,,\n")

    # finalize override report
    if orp:
        orp.close()

    # Final summary appended
    try:
        with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
            log_file.write("\nFinal Part -> Assigned Name mapping (summary):\n")
            for raw_key in ordered_parts:
                cnt = stats[raw_key]["count"]
                assigned = name_map.get(raw_key, raw_key)
                log_file.write(f"  {raw_key:<12} ({cnt:>4} voxels) => {sanitize_filename(assigned)}\n")
    except Exception:
        pass

    print(f"Export complete for {vox_file}. See {export_log_path} and {override_report_path} for details.")