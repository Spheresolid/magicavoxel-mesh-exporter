import os
import csv
import json
import argparse
import subprocess
import sys
import numpy as np
from Vox200Parser import Vox200Parser

BASE_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
APPLY_SCRIPT = os.path.join(BASE_DIR, "ApplyOverrides.py")

def read_finalmapping(vox):
    csv_path = os.path.join(REPORTS_DIR, f"FinalMapping_{vox}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FinalMapping CSV not found: {csv_path}")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

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
    return suggestions, metrics, max_dist

def compute_tiny_to_large(rows, vox_basename, tiny_abs=50, tiny_frac=0.05, size_ratio=4.0, iou_thresh=0.05, centroid_thresh=8.0):
    """
    Heuristic:
    - If a named Part is very small (<= min(tiny_abs, tiny_frac * largest_part))
      and a different Part is much larger (size_ratio >= size_ratio) and close
      by centroid or has AABB IoU, suggest remapping the name to that larger Part.
    Returns dict raw_large -> intended and metrics for those suggestions.
    """
    heur_suggestions = {}
    heur_metrics = {}

    # Load .vox to get per-part centroids, counts and AABBs
    vox_path = os.path.join(BASE_DIR, f"{vox_basename}.vox")
    if not os.path.exists(vox_path):
        return heur_suggestions, heur_metrics

    parser = Vox200Parser(vox_path).parse()
    voxels_by_layer = parser.voxels_by_layer     # dict 'Part_X' -> list of Voxels
    # build arrays
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

    # build a quick lookup of intended names from FinalMapping rows
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
        # only consider tiny parts that carry an intended name and where the exported file differed
        if small_count > tiny_threshold:
            continue

        # try find candidate larger parts
        candidates = []
        for p_large, large_count in counts.items():
            if p_large == raw_small:
                continue
            if large_count < 1:
                continue
            size_ratio_actual = float(large_count) / float(max(1, small_count))
            if size_ratio_actual < size_ratio:
                continue
            # compute centroid distance and IoU
            c_small = centroids.get(raw_small)
            c_large = centroids.get(p_large)
            if c_small is None or c_large is None:
                continue
            centroid_dist = float(np.linalg.norm(c_small - c_large))
            min_a, max_a = aabbs.get(raw_small, (None, None))
            min_b, max_b = aabbs.get(p_large, (None, None))
            # compute IoU safely
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

        # pick best candidate by smallest centroid_dist then highest IoU (prioritize overlap)
        if not candidates:
            continue
        # prefer IoU >= threshold first, else fallback to closest centroid
        iou_candidates = [c for c in candidates if c[4] >= iou_thresh]
        if iou_candidates:
            # pick candidate with largest IoU then largest size ratio
            best = sorted(iou_candidates, key=lambda x: (-x[4], -x[2]))[0]
        else:
            # pick by centroid distance threshold
            close_candidates = [c for c in candidates if c[3] <= centroid_thresh]
            if close_candidates:
                best = sorted(close_candidates, key=lambda x: (x[3], -x[2]))[0]
            else:
                # none meet IoU or centroid proximity -> skip
                continue

        p_large, large_count, size_ratio_actual, centroid_dist, iou = best
        # If the large part already intends to the same name, skip (no change)
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
    return high

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def run_apply(vox, overrides_path, commit=False):
    if not os.path.exists(APPLY_SCRIPT):
        print(f"Apply script not found: {APPLY_SCRIPT}")
        return 2
    cmd = [sys.executable, APPLY_SCRIPT, "--vox", vox, "--overrides", overrides_path]
    if commit:
        cmd.append("--commit")
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode

def write_suggestion_text(path, suggestions, metrics, heur_metrics):
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

def main():
    p = argparse.ArgumentParser(description="Auto-generate and optionally apply overrides from FinalMapping.")
    p.add_argument("--vox", required=True, help="vox basename (e.g. Character)")
    # Tuned defaults: conservative but catches big misassignments (like Head on a large part)
    p.add_argument("--dist-threshold", type=float, default=0.15, help="normalized centroid distance threshold for high-confidence (default 0.15)")
    p.add_argument("--min-voxels", type=int, default=20, help="minimum voxel count for high-confidence (default 20)")
    p.add_argument("--auto", action="store_true", help="apply high-confidence overrides automatically (commit)")
    p.add_argument("--preview-only", action="store_true", help="generate files and run ApplyOverrides dry-run only")
    # tiny->large heuristic tuning
    p.add_argument("--enable-tiny2large", action="store_true", help="enable conservative tiny->large remapping heuristic")
    p.add_argument("--tiny-abs", type=int, default=50, help="absolute max voxel count to consider a part 'tiny' (default 50)")
    p.add_argument("--tiny-frac", type=float, default=0.05, help="fraction of largest part to consider as tiny (default 0.05)")
    p.add_argument("--size-ratio", type=float, default=4.0, help="minimum larger/smaller size ratio to consider remap (default 4.0)")
    p.add_argument("--iou-threshold", type=float, default=0.05, help="IoU threshold for heuristic (default 0.05)")
    p.add_argument("--centroid-threshold", type=float, default=8.0, help="centroid distance (voxels) threshold for heuristic (default 8)")
    args = p.parse_args()

    try:
        rows = read_finalmapping(args.vox)
    except FileNotFoundError as e:
        print(e)
        return 1

    suggestions, metrics, max_dist = build_suggestions(rows)
    if not suggestions:
        print("No mismatches found; nothing to suggest.")
        return 0

    # optionally compute tiny->large heuristic suggestions
    heur_suggestions = {}
    heur_metrics = {}
    if args.enable_tiny2large:
        heur_suggestions, heur_metrics = compute_tiny_to_large(
            rows,
            args.vox,
            tiny_abs=args.tiny_abs,
            tiny_frac=args.tiny_frac,
            size_ratio=args.size_ratio,
            iou_thresh=args.iou_threshold,
            centroid_thresh=args.centroid_threshold
        )
        if heur_suggestions:
            # merge heuristic into suggestions (heuristic targets are larger parts -> intended)
            # avoid overwriting existing suggestion for the same raw if present
            for raw, intended in heur_suggestions.items():
                if raw not in suggestions:
                    suggestions[raw] = intended
            # incorporate heur metrics into metrics dict
            for raw, mm in heur_metrics.items():
                metrics.setdefault(raw, {})
                metrics[raw].update(mm)

    # write per-vox JSONs into reports/
    SUGGEST_JSON = os.path.join(REPORTS_DIR, f"name_overrides_suggested_{args.vox}.json")
    HIGHCONF_JSON = os.path.join(REPORTS_DIR, f"name_overrides_highconf_{args.vox}.json")
    HEUR_JSON = os.path.join(REPORTS_DIR, f"name_overrides_heur_suggested_{args.vox}.json")
    SUGGEST_TXT = os.path.join(REPORTS_DIR, f"name_overrides_suggested_{args.vox}.txt")

    write_json(SUGGEST_JSON, suggestions)
    print(f"Wrote suggestions: {SUGGEST_JSON}  (total {len(suggestions)} entries)")

    # write a separate heur file for easy inspection
    if heur_suggestions:
        write_json(HEUR_JSON, heur_suggestions)
        print(f"Wrote heuristic-only suggestions: {HEUR_JSON}  (count {len(heur_suggestions)})")

    # human readable suggestions text
    write_suggestion_text(SUGGEST_TXT, suggestions, metrics, heur_metrics)
    print(f"Wrote human-readable suggestions: {SUGGEST_TXT}")

    high = filter_highconf(suggestions, metrics, max_dist, args.dist_threshold, args.min_voxels)
    write_json(HIGHCONF_JSON, high)
    print(f"Wrote high-confidence overrides: {HIGHCONF_JSON}  (count {len(high)})")

    # human-readable summary
    print("\nHigh-confidence overrides (raw -> intended):")
    for k, v in sorted(high.items()):
        m = metrics.get(k, {})
        dist = m.get("CentroidDistance")
        vox = m.get("VoxelCount")
        print(f"  {k} -> {v}   (dist={dist:.6g}, vox={vox})")

    # heuristic summary (if any)
    if heur_suggestions:
        print("\nTiny->Large heuristic suggestions (large_raw -> intended) with metrics:")
        for k, v in sorted(heur_suggestions.items()):
            m = heur_metrics.get(k, {})
            sf = m.get("SuggestedFrom")
            sr = m.get("SizeRatio")
            cd = m.get("CentroidDistanceParts")
            iou = m.get("IoU_parts")
            sc = m.get("SmallCount")
            lc = m.get("LargeCount")
            print(f"  {k} -> {v}  (from {sf}, small={sc}, large={lc}, ratio={sr:.2f}, cd={cd:.2f}, iou={iou:.3g})")

    # run ApplyOverrides dry-run for highconf
    if not high:
        print("\nNo high-confidence overrides to apply automatically.")
        return 0

    rc = run_apply(args.vox, HIGHCONF_JSON, commit=False)
    if rc != 0:
        print("ApplyOverrides dry-run failed (check output). Aborting auto-commit.")
        return rc

    if args.auto:
        print("\nAuto-commit enabled. Performing commit now.")
        rc = run_apply(args.vox, HIGHCONF_JSON, commit=True)
        if rc == 0:
            print("Auto-commit complete. Backups and manifest written to reports/.")
            print("If you need to undo: python UndoRenames.py --vox", args.vox, "--commit")
        else:
            print("Auto-commit failed. Inspect reports/ for details.")
        return rc
    else:
        print("\nDry-run complete. If you agree with the plan, run:")
        print(f"  python {os.path.basename(__file__)} --vox {args.vox} --auto --enable-tiny2large")
        print("or commit with ApplyOverrides.py manually:\n  python ApplyOverrides.py --vox", args.vox, "--overrides", os.path.basename(HIGHCONF_JSON), "--commit")
    return 0

if __name__ == "__main__":
    sys.exit(main())