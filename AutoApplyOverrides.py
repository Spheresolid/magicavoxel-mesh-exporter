#!/usr/bin/env python3
import os
import csv
import json
import argparse
import subprocess
import sys
import numpy as np
from Vox200Parser import Vox200Parser

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
APPLY_SCRIPT = os.path.join(BASE_DIR, "ApplyOverrides.py")
FINALIZE_SCRIPT = os.path.join(BASE_DIR, "FinalizeMapping.py")
DEBUG_LOG = os.path.join(REPORTS_DIR, "autoapply_debug.log")

def dbg(msg):
    line = f"[AutoApply] {msg}"
    print(line)
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def read_finalmapping(vox):
    csv_path = os.path.join(REPORTS_DIR, f"FinalMapping_{vox}.csv")
    dbg(f"Attempting to read FinalMapping CSV at: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FinalMapping CSV not found: {csv_path}")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    dbg(f"Read {len(rows)} rows from FinalMapping CSV")
    return rows

def try_generate_finalmapping(vox):
    csv_path = os.path.join(REPORTS_DIR, f"FinalMapping_{vox}.csv")
    if os.path.exists(csv_path):
        dbg("FinalMapping already exists; no generation needed.")
        return True
    if not os.path.exists(FINALIZE_SCRIPT):
        dbg(f"FinalizeMapping.py not found at {FINALIZE_SCRIPT}; cannot auto-generate FinalMapping.")
        return False
    dbg(f"FinalMapping for {vox} missing — running FinalizeMapping.py to generate it.")
    proc = subprocess.run([sys.executable, FINALIZE_SCRIPT], cwd=BASE_DIR)
    dbg(f"FinalizeMapping.py exited with code {proc.returncode}")
    exists = os.path.exists(csv_path)
    dbg(f"FinalMapping exists after finalize attempt: {exists} -> {csv_path}")
    return exists

def basename_noext(path):
    return os.path.splitext(os.path.basename(path))[0] if path else ""

def build_suggestions(rows):
    suggestions = {}
    metrics = {}
    max_dist = 0.0
    for r in rows:
        export = (r.get("ExportFile") or "").strip()
        raw = (r.get("RawPart") or "").strip()
        intended = (r.get("IntendedName") or "").strip()
        dist = r.get("CentroidDistance", "")
        try:
            distv = float(dist) if dist not in (None, "") else float("inf")
        except Exception:
            distv = float("inf")
        voxcount = r.get("VoxelCount")
        try:
            vox = int(voxcount) if voxcount not in (None, "") else 0
        except Exception:
            vox = 0
        if not raw:
            continue
        exp_base = basename_noext(export)
        # Suggest only when exported filename != intended name
        if exp_base != intended and intended:
            suggestions[raw] = intended
            metrics[raw] = {"ExportFile": os.path.basename(export), "Intended": intended, "CentroidDistance": distv, "VoxelCount": vox}
            if distv != float("inf") and distv > max_dist:
                max_dist = distv
    dbg(f"Built {len(suggestions)} initial suggestions, max_dist={max_dist}")
    return suggestions, metrics, max_dist

def compute_tiny_to_large(rows, vox_basename, tiny_abs=50, tiny_frac=0.05, size_ratio=4.0, iou_thresh=0.05, centroid_thresh=8.0):
    heur_suggestions = {}
    heur_metrics = {}

    vox_path = os.path.join(BASE_DIR, f"{vox_basename}.vox")
    if not os.path.exists(vox_path):
        dbg(f"No .vox file found at {vox_path}; skipping tiny2large heuristic")
        return heur_suggestions, heur_metrics

    parser = Vox200Parser(vox_path).parse()
    voxels_by_layer = parser.voxels_by_layer
    parts = [k for k in voxels_by_layer.keys() if voxels_by_layer[k]]
    if not parts:
        return heur_suggestions, heur_metrics

    centroids = {}
    counts = {}
    aabbs = {}
    for p in parts:
        voxels = voxels_by_layer[p]
        pts = np.array([[v.x, v.y, v.z] for v in voxels], dtype=float)
        centroids[p] = pts.mean(axis=0)
        counts[p] = pts.shape[0]
        aabbs[p] = (pts.min(axis=0), pts.max(axis=0))

    largest_count = max(counts.values()) if counts else 0
    tiny_threshold = min(tiny_abs, int(max(1, tiny_frac * largest_count)))

    intended_by_raw = {}
    for r in rows:
        raw = (r.get("RawPart") or "").strip()
        intended = (r.get("IntendedName") or "").strip()
        if raw and intended:
            intended_by_raw[raw] = intended

    for raw_small, intended in intended_by_raw.items():
        small_count = counts.get(raw_small, 0)
        if small_count <= 0:
            continue
        if small_count > tiny_threshold:
            continue

        candidates = []
        for p_large, large_count in counts.items():
            if p_large == raw_small:
                continue
            if large_count < 1:
                continue
            size_ratio_actual = float(large_count) / float(max(1, small_count))
            if size_ratio_actual < size_ratio:
                continue
            c_small = centroids.get(raw_small)
            c_large = centroids.get(p_large)
            if c_small is None or c_large is None:
                continue
            centroid_dist = float(np.linalg.norm(c_small - c_large))
            min_a, max_a = aabbs.get(raw_small, (None, None))
            min_b, max_b = aabbs.get(p_large, (None, None))
            if min_a is None or min_b is None:
                iou = 0.0
            else:
                inter_min = np.maximum(min_a, min_b)
                inter_max = np.minimum(max_a, max_b)
                inter_dims = np.maximum(inter_max - inter_min, 0.0)
                inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])
                vol_a = float(np.prod(np.maximum(max_a - min_a, 0.0)))
                vol_b = float(np.prod(np.maximum(max_b - min_b, 0.0)))
                union = vol_a + vol_b - inter_vol
                iou = float(inter_vol / union) if union > 0.0 else 0.0

            candidates.append((p_large, large_count, size_ratio_actual, centroid_dist, iou))

        if not candidates:
            continue
        iou_candidates = [c for c in candidates if c[4] >= iou_thresh]
        if iou_candidates:
            best = sorted(iou_candidates, key=lambda x: (-x[4], -x[2]))[0]
        else:
            close_candidates = [c for c in candidates if c[3] <= centroid_thresh]
            if close_candidates:
                best = sorted(close_candidates, key=lambda x: (x[3], -x[2]))[0]
            else:
                continue

        p_large, large_count, size_ratio_actual, centroid_dist, iou = best
        if intended_by_raw.get(p_large) == intended:
            continue

        heur_suggestions[p_large] = intended
        heur_metrics[p_large] = {
            "SuggestedFrom": raw_small,
            "Intended": intended,
            "SmallCount": small_count,
            "LargeCount": large_count,
            "SizeRatio": size_ratio_actual,
            "CentroidDistanceParts": centroid_dist,
            "IoU_parts": iou
        }

    dbg(f"Built {len(heur_suggestions)} heuristic suggestions")
    return heur_suggestions, heur_metrics

def filter_highconf(suggestions, metrics, max_dist, dist_threshold, min_voxels):
    high = {}
    if max_dist == 0.0:
        max_dist = 1.0
    for raw, intended in suggestions.items():
        m = metrics.get(raw, {})
        dist = m.get("CentroidDistance", float("inf"))
        vox = m.get("VoxelCount", 0)
        norm = dist / max_dist if dist != float("inf") else float("inf")
        if norm <= dist_threshold and vox >= min_voxels:
            high[raw] = intended
    dbg(f"Filtered {len(high)} high-confidence suggestions")
    return high

def write_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        dbg(f"Wrote JSON: {path}  (len={len(obj) if isinstance(obj, dict) else 'N/A'})")
    except Exception as e:
        dbg(f"ERROR writing JSON {path}: {e}")

def run_apply(vox, overrides_path, commit=False):
    if not os.path.exists(APPLY_SCRIPT):
        dbg(f"Apply script not found: {APPLY_SCRIPT}")
        return 2
    cmd = [sys.executable, APPLY_SCRIPT, "--vox", vox, "--overrides", overrides_path]
    if commit:
        cmd.append("--commit")
    dbg("Running: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    dbg(f"ApplyOverrides exited {proc.returncode}")
    return proc.returncode

def write_suggestion_text(path, suggestions, metrics, heur_metrics):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("AutoApplyOverrides suggestions\n\n")
            f.write("Suggested overrides (raw -> intended):\n")
            for raw, intended in sorted(suggestions.items()):
                m = metrics.get(raw, {})
                f.write(f"  {raw} -> {intended}  (ExportFile={m.get('ExportFile')}, CentroidDistance={m.get('CentroidDistance')}, Voxels={m.get('VoxelCount')})\n")
            if heur_metrics:
                f.write("\nHeuristic tiny->large suggestions (large_raw -> intended):\n")
                for large, mm in sorted(heur_metrics.items()):
                    f.write(f"  {large} -> {mm.get('Intended')}  (from {mm.get('SuggestedFrom')}, small={mm.get('SmallCount')}, large={mm.get('LargeCount')}, ratio={mm.get('SizeRatio'):.2f}, cd={mm.get('CentroidDistanceParts'):.2f}, iou={mm.get('IoU_parts'):.3g})\n")
        dbg(f"Wrote suggestions text: {path}")
    except Exception as e:
        dbg(f"ERROR writing suggestions text {path}: {e}")

def main():
    p = argparse.ArgumentParser(description="Auto-generate and optionally apply overrides from FinalMapping.")
    p.add_argument("--vox", required=True, help="vox basename (e.g. Character)")
    p.add_argument("--dist-threshold", type=float, default=0.15)
    p.add_argument("--min-voxels", type=int, default=20)
    p.add_argument("--auto", action="store_true")
    p.add_argument("--enable-tiny2large", action="store_true")
    args = p.parse_args()

    dbg(f"BASE_DIR={BASE_DIR} REPORTS_DIR={REPORTS_DIR} DEBUG_LOG={DEBUG_LOG}")

    rows = []
    try:
        rows = read_finalmapping(args.vox)
    except FileNotFoundError:
        dbg(f"FinalMapping_{args.vox}.csv not found; attempting to run FinalizeMapping.py")
        if os.path.exists(FINALIZE_SCRIPT):
            proc = subprocess.run([sys.executable, FINALIZE_SCRIPT], cwd=BASE_DIR)
            dbg(f"FinalizeMapping.py exit code {proc.returncode}")
            try:
                rows = read_finalmapping(args.vox)
            except FileNotFoundError:
                dbg("FinalizeMapping did not produce FinalMapping CSV; continuing with empty rows.")
                rows = []
        else:
            dbg("FinalizeMapping.py not present; continuing with empty rows.")
            rows = []

    suggestions, metrics, max_dist = build_suggestions(rows)

    heur_suggestions = {}
    heur_metrics = {}
    if args.enable_tiny2large:
        heur_suggestions, heur_metrics = compute_tiny_to_large(
            rows,
            args.vox,
            tiny_abs=50,
            tiny_frac=0.05,
            size_ratio=4.0,
            iou_thresh=0.05,
            centroid_thresh=8.0
        )
        if heur_suggestions:
            for raw, intended in heur_suggestions.items():
                if raw not in suggestions:
                    suggestions[raw] = intended
            for raw, mm in heur_metrics.items():
                metrics.setdefault(raw, {})
                metrics[raw].update(mm)

    SUGGEST_JSON = os.path.join(REPORTS_DIR, f"name_overrides_suggested_{args.vox}.json")
    HIGHCONF_JSON = os.path.join(REPORTS_DIR, f"name_overrides_highconf_{args.vox}.json")
    HEUR_JSON = os.path.join(REPORTS_DIR, f"name_overrides_heur_suggested_{args.vox}.json")
    SUGGEST_TXT = os.path.join(REPORTS_DIR, f"name_overrides_suggested_{args.vox}.txt")

    # Always write suggestion files (may be empty dicts)
    write_json(SUGGEST_JSON, suggestions)
    write_json(HEUR_JSON, heur_suggestions if heur_suggestions else {})
    write_suggestion_text(SUGGEST_TXT, suggestions, metrics, heur_metrics)
    write_json(HIGHCONF_JSON, filter_highconf(suggestions, metrics, max_dist, args.dist_threshold, args.min_voxels))

    dbg("AutoApplyOverrides finished normal execution.")
    return 0

if __name__ == "__main__":
    sys.exit(main())