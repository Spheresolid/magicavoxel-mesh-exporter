# ExportMeshes.py
# -*- coding: utf-8 -*-
"""
ExportMeshes.py
Exports per-Part meshes from .vox using Vox200Parser, applies per-model transforms (if present),
and optionally remaps axes (preset --to-maya uses "x,z,-y").

Key safety features:
 - Validate rotation/translation attrs before applying.
 - Clamp/ignore invalid transforms (NaN/Inf or enormous translations).
 - Normalize numeric precision and remove degenerate geometry before export.
 - Deterministic OBJ writer (write_stable_obj) used; falls back to trimesh exporter on unexpected geometry.
 - Final Maya compatibility export step to ensure vertex normals and deterministic OBJ layout.
 - Optionally write hard-edged OBJs (per-face duplicated vertices + per-face normals) so Maya gets flat shading.
"""
import os
import sys
import math
import traceback
import csv
import glob
import re
import argparse

import numpy as np
import trimesh
from trimesh.transformations import quaternion_matrix, euler_matrix

from Vox200Parser import Vox200Parser

# CLI
cli = argparse.ArgumentParser(description="Export .vox parts as .obj with optional per-model transforms and axes remap.")
cli.add_argument("--to-maya", action="store_true", help="Remap exported geometry into Maya's world (preset axes map).")
cli.add_argument("--axes-map", help="Custom axes map, e.g. 'x,z,-y' meaning new=(x, z, -y).")
cli.add_argument("--vox", help="Optional: limit to a single .vox basename (no extension).")
cli.add_argument("--debug", action="store_true", help="Emit debug info")
cli.add_argument("--hard-edges", action="store_true", help="Write hard-edged OBJ (duplicate vertices per face) for flat shading (Maya-safe).")
args = cli.parse_args()

# Pick hard edges automatically when exporting for Maya, or when user explicitly requested it.
HARD_EDGES = bool(args.hard_edges) or bool(args.to_maya)

# Preset
AXES_MAP_SPEC = None
if args.to_maya:
    AXES_MAP_SPEC = "x,z,-y"
elif args.axes_map:
    AXES_MAP_SPEC = args.axes_map

def sanitize_filename(name):
    if not name:
        return ""
    s = str(name)
    safe = "".join(c if (c.isalnum() or c in (' ', '.', '_', '-')) else '_' for c in s)
    while '__' in safe:
        safe = safe.replace('__', '_')
    safe = safe.strip().replace(' ', '_')
    return safe or ""

def compute_centroid(voxels):
    pts = np.array([[v.x, v.y, v.z] for v in voxels], dtype=float)
    if pts.size == 0:
        return np.array([math.nan, math.nan, math.nan], dtype=float)
    return pts.mean(axis=0)

def parse_axes_map(spec):
    if not spec:
        return None
    toks = [t.strip().lower() for t in spec.split(",")]
    if len(toks) != 3:
        raise ValueError("axes-map must have three comma-separated tokens, e.g. x,y,z or x,z,-y")
    mat = np.zeros((3,3), dtype=float)
    axis_idx = {"x":0, "y":1, "z":2}
    for i, tok in enumerate(toks):
        sign = -1.0 if tok.startswith("-") else 1.0
        t = tok[1:] if tok.startswith("-") else tok
        if t not in axis_idx:
            raise ValueError("invalid axis token in axes-map: " + tok)
        mat[i, axis_idx[t]] = sign
    return mat

def write_stable_obj(path, mesh):
    """
    Deterministic OBJ writer:
    - vertices 'v x y z' (6 decimals)
    - normals 'vn x y z' (6 decimals) when available
    - faces as triangles using v//vn if normals present, otherwise v v v
    If faces are non-triangles, fall back to trimesh export.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        # fallback: use trimesh writer (explicit file_type)
        mesh.export(path, file_type='obj')
        return

    vns = None
    try:
        if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
            vns = np.asarray(mesh.vertex_normals, dtype=np.float64)
            if vns.shape[0] != verts.shape[0]:
                vns = None
    except Exception:
        vns = None

    fmt = "{:.6f} {:.6f} {:.6f}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Stable OBJ exported by ExportMeshes.py\n")
        for v in verts:
            f.write("v " + fmt.format(float(v[0]), float(v[1]), float(v[2])))
        if vns is not None:
            for n in vns:
                f.write("vn " + fmt.format(float(n[0]), float(n[1]), float(n[2])))
        for face in faces:
            a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
            if vns is not None:
                f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                f.write(f"f {a} {b} {c}\n")

def write_hard_obj(path, mesh):
    """
    Write an OBJ with hard edges:
    - duplicate vertex coordinates per face so vertices are not shared
    - write a face-normal (vn) per vertex (all three the same) so faces are flat
    - write 's off' to disable smoothing groups
    This produces flat-shaded OBJ and explicit vn per-face.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=int)
    fmt = "{:.6f} {:.6f} {:.6f}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Hard-edged OBJ exported by ExportMeshes.py\n")
        f.write("s off\n")
        idx = 1
        for face in faces:
            # Defensive: ensure indices are ints and in-range
            a_idx = int(face[0])
            b_idx = int(face[1])
            c_idx = int(face[2])
            if a_idx < 0 or b_idx < 0 or c_idx < 0:
                # negative indices are unexpected here — fall back to stable writer elsewhere
                raise RuntimeError("write_hard_obj encountered negative index")
            v0 = verts[a_idx]
            v1 = verts[b_idx]
            v2 = verts[c_idx]
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if not np.isfinite(norm) or norm == 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                n = n / norm
            # write three vertex entries (duplicate per-face)
            f.write("v " + fmt.format(float(v0[0]), float(v0[1]), float(v0[2])))
            f.write("v " + fmt.format(float(v1[0]), float(v1[1]), float(v1[2])))
            f.write("v " + fmt.format(float(v2[0]), float(v2[1]), float(v2[2])))
            # write corresponding normals (one per vertex, same normal)
            f.write("vn " + fmt.format(float(n[0]), float(n[1]), float(n[2])))
            f.write("vn " + fmt.format(float(n[0]), float(n[1]), float(n[2])))
            f.write("vn " + fmt.format(float(n[0]), float(n[1]), float(n[2])))
            # face referencing v//vn indices
            f.write(f"f {idx}//{idx} {idx+1}//{idx+1} {idx+2}//{idx+2}\n")
            idx += 3

def apply_maya_compatibility_export(mesh, export_path, debug=False, hard_edges=False):
    """
    Ultimate cleanup for Maya compatibility:
     - deep process (trimesh.process) to clean geometry
     - repair (remove degenerate/duplicate faces, unreferenced verts)
     - triangulate if required
     - compute/fix vertex normals
     - write deterministic OBJ via write_stable_obj or write_hard_obj (atomic replace)
    Returns True on success.
    """
    try:
        m = mesh.copy()

        # 1) Deep internal cleanup if available
        try:
            if hasattr(m, "process"):
                m.process()
        except Exception:
            if debug:
                print("[MAYA_FIX] deep process() failed or not available; continuing")

        # 2) Best-effort repairs
        for fn in ("remove_degenerate_faces", "remove_duplicate_faces", "remove_unreferenced_vertices"):
            try:
                if hasattr(m, fn):
                    getattr(m, fn)()
            except Exception:
                pass

        # 3) Triangulate if faces are polygons
        try:
            fa = np.asarray(getattr(m, "faces", np.array([])), dtype=int)
            if fa.size == 0:
                if debug:
                    print("[MAYA_FIX] Mesh has no faces; skipping export")
                return False
            if fa.ndim != 2 or fa.shape[1] != 3:
                try:
                    mtri = m.copy().triangulate()
                    if isinstance(mtri, trimesh.Trimesh) and getattr(mtri, "faces", None) is not None:
                        m = mtri
                except Exception:
                    if debug:
                        print("[MAYA_FIX] triangulate() failed; will fallback to trimesh exporter")
        except Exception:
            if debug:
                print("[MAYA_FIX] face inspection failed; will fallback to trimesh exporter")

        # 4) Compute/fix normals
        try:
            if hasattr(m, "fix_normals"):
                m.fix_normals()
            try:
                _ = m.vertex_normals
            except Exception:
                if debug:
                    print("[MAYA_FIX] vertex_normals access failed")
        except Exception:
            if debug:
                print("[MAYA_FIX] fix_normals failed; continuing")

        # 5) Reduce precision to stable values
        try:
            verts = np.asarray(m.vertices, dtype=np.float64)
            verts = np.round(verts, 6)
            m.vertices = verts.astype(np.float64)
        except Exception:
            pass

        # 6) Write deterministic OBJ to a .tmp.obj and atomically replace
        tmp = export_path + ".tmp.obj"
        try:
            if hard_edges:
                write_hard_obj(tmp, m)
            else:
                write_stable_obj(tmp, m)
            # final defensive pass: strip mtllib/usemtl if any
            with open(tmp, "r", encoding="utf-8", errors="ignore") as f_in:
                lines = [L for L in f_in.readlines() if not (L.strip().lower().startswith("mtllib") or L.strip().lower().startswith("usemtl"))]
            with open(tmp, "w", encoding="utf-8", errors="ignore") as f_out:
                f_out.writelines(lines)
            os.replace(tmp, export_path)
            return True
        except Exception as ex:
            if debug:
                print(f"[MAYA_FIX] primary writer failed: {ex}; trying trimesh.export fallback")
            # If hard edges were requested, try hard writer as fallback if primary wasn't hard
            try:
                fallback_tmp = export_path + ".tmp.obj"
                if hard_edges:
                    # try hard writer explicitly (may raise)
                    write_hard_obj(fallback_tmp, m)
                    with open(fallback_tmp, "r", encoding="utf-8", errors="ignore") as f_in:
                        lines = [L for L in f_in.readlines() if not (L.strip().lower().startswith("mtllib") or L.strip().lower().startswith("usemtl"))]
                    with open(export_path, "w", encoding="utf-8", errors="ignore") as f_out:
                        f_out.writelines(lines)
                    try:
                        os.remove(fallback_tmp)
                    except Exception:
                        pass
                    return True
                # last resort: let trimesh export explicit OBJ and we clean it
                m.export(fallback_tmp, file_type='obj')
                with open(fallback_tmp, "r", encoding="utf-8", errors="ignore") as f_in:
                    lines = [L for L in f_in.readlines() if not (L.strip().lower().startswith("mtllib") or L.strip().lower().startswith("usemtl"))]
                with open(export_path, "w", encoding="utf-8", errors="ignore") as f_out:
                    f_out.writelines(lines)
                try:
                    os.remove(fallback_tmp)
                except Exception:
                    pass
                return True
            except Exception as ex2:
                if debug:
                    print(f"[MAYA_FIX] fallback failed: {ex2}")
                return False

    except Exception as fatal:
        if debug:
            print(f"[MAYA_FIX] Unexpected error: {fatal}")
        return False

# Directories
BASE_DIR = os.path.dirname(__file__)
os.chdir(BASE_DIR)
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
os.makedirs(EXPORT_ROOT, exist_ok=True)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# remap matrix
remap_R = None
if AXES_MAP_SPEC:
    try:
        remap_R = parse_axes_map(AXES_MAP_SPEC)
    except Exception as ex:
        print(f"Invalid axes map '{AXES_MAP_SPEC}': {ex}")
        remap_R = None

def load_reference_mappings(vox_name):
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
    dbg_path = os.path.join(REPORTS_DIR, f"DebugNameMap_{vox_name}.txt")
    if os.path.exists(dbg_path):
        try:
            with open(dbg_path, "r", encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Part_") and "=>" in line:
                        try:
                            left, right = line.split("=>", 1)
                            left = left.strip()
                            right = right.strip()
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
    if count is None:
        return None
    matches = []
    for e in refs:
        ec = e.get("count")
        if ec is None:
            continue
        diff = abs(ec - count)
        rel = diff / max(1.0, ec)
        if diff <= tol_abs or rel <= tol_rel:
            matches.append((diff, rel, e))
    if not matches:
        return None
    matches.sort(key=lambda x: (x[0], x[1]))
    if len(matches) > 1 and matches[0][0] == matches[1][0] and matches[0][1] == matches[1][1]:
        return None
    return matches[0][2]

# Main export loop
vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]
if args.vox:
    target = f"{args.vox}.vox"
    if target in vox_files:
        vox_files = [target]
    else:
        print(f"Specified vox not found: {args.vox}")
        vox_files = []

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

    export_log_path = os.path.join(sub_export_dir, "ExportLog.txt")
    override_report_path = os.path.join(REPORTS_DIR, f"VoxelOverrideReport_{vox_name}.txt")
    layer_map_path = os.path.join(REPORTS_DIR, f"LayerMapping_{vox_name}.txt")

    try:
        with open(layer_map_path, "w", encoding="utf-8", errors="replace") as lm:
            lm.write(f"LayerMapping for: {vox_file}\n\n")
            lm.write(f"{'Part':<12} {'VoxelCount':>10} {'ParserAssigned':<30}\n")
            lm.write(f"{'-'*12} {'-'*10} {'-'*30}\n")
            for raw_key in ordered_parts:
                assigned = name_map.get(raw_key, raw_key)
                cnt = stats[raw_key]["count"]
                lm.write(f"{raw_key:<12} {cnt:10d} {str(assigned):<30}\n")
    except Exception:
        pass

    try:
        with open(export_log_path, "w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"Export Status & Layer Name Mapping for {vox_file}\n\n")
    except Exception:
        continue

    try:
        orp = open(override_report_path, "w", encoding="utf-8", errors="replace")
        orp.write(f"Voxel Override Report for: {vox_file}\n\n")
        orp.write("Part, VoxelCount, Centroid, ParserName, FinalName, NameSource, Reason\n")
    except Exception:
        orp = None

    used_filenames = set()

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

        if refs:
            best_ref = find_best_ref_by_count(refs, voxel_count, tol_abs=3, tol_rel=0.02)
            if best_ref:
                if best_ref["raw"] != raw_key:
                    final_name = best_ref["name"]
                    name_source = "finalmapping_override"
                    reason = f"count_match (ref {best_ref['raw']} count={best_ref.get('count')})"
                else:
                    final_name = best_ref["name"] or parser_assigned
                    name_source = "finalmapping_confirm"
                    reason = "ref_confirm_same_raw"
            else:
                if (not parser_assigned) or (parser_assigned == raw_key) or parser_assigned.startswith("_"):
                    fuzzy = find_best_ref_by_count(refs, voxel_count, tol_abs=max(3, int(0.01*voxel_count)), tol_rel=0.05)
                    if fuzzy and fuzzy["raw"] != raw_key:
                        final_name = fuzzy["name"]
                        name_source = "finalmapping_fuzzy"
                        reason = f"fuzzy_count_match (ref {fuzzy['raw']} count={fuzzy.get('count')})"

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

        try:
            positions = np.array([[v.x, v.y, v.z] for v in voxels])
            cubes = []
            for pos in positions:
                cube = trimesh.creation.box(extents=(1, 1, 1))
                cube.apply_translation(pos)
                cubes.append(cube)

            if cubes:
                combined = trimesh.util.concatenate(cubes)

                # Apply per-model transform if present in parser.model_transforms
                model_idx = None
                try:
                    model_idx = int(raw_key.split("_",1)[1])
                except Exception:
                    model_idx = None

                attrs = getattr(parser, "model_transforms", {}).get(model_idx) if model_idx is not None else None
                # --- Validate & apply per-model transform safely ---
                if attrs:
                    # parse translation
                    tvals = None
                    if "_t" in attrs:
                        toks = [tok for tok in re.split(r"[,\s]+", attrs["_t"].strip()) if tok != ""]
                        try:
                            if len(toks) >= 3:
                                tvals = [float(toks[0]), float(toks[1]), float(toks[2])]
                        except Exception:
                            tvals = None

                    # parse rotation into a 4x4 matrix (quat [w,x,y,z] or euler deg triplet)
                    rot_mat = None
                    if "_r" in attrs:
                        toks = [tok for tok in re.split(r"[,\s]+", attrs["_r"].strip()) if tok != ""]
                        try:
                            nums = [float(x) for x in toks]
                            if len(nums) == 4:
                                # try quaternion; ensure finite
                                try:
                                    candidate = quaternion_matrix(nums)
                                except Exception:
                                    # try alternate ordering [x,y,z,w]
                                    candidate = quaternion_matrix([nums[3], nums[0], nums[1], nums[2]])
                                    rot_mat = candidate
                            elif len(nums) == 3:
                                rx, ry, rz = [math.radians(a) for a in nums]
                                rot_mat = euler_matrix(rx, ry, rz, axes='sxyz')
                        except Exception:
                            rot_mat = None

                    # Validate rotation matrix (finite and non-singular)
                    if rot_mat is not None:
                        try:
                            R3 = rot_mat[:3, :3].astype(float)
                            if not np.isfinite(R3).all() or abs(np.linalg.det(R3)) < 1e-8:
                                if args and getattr(args, "debug", False):
                                    print(f"[DEBUG] Invalid rotation matrix for {raw_key}; ignoring rotation.")
                                rot_mat = None
                        except Exception:
                            rot_mat = None

                    # Validate translation: finite and within sane bounds
                    MAX_T = 1e5
                    if tvals is not None:
                        try:
                            tv = np.array(tvals, dtype=float)
                            if not np.isfinite(tv).all() or np.any(np.abs(tv) > MAX_T):
                                if args and getattr(args, "debug", False):
                                    print(f"[DEBUG] Translation {tvals} out-of-range for {raw_key}; ignoring translation.")
                                tvals = None
                            else:
                                tvals = tv.tolist()
                        except Exception:
                            tvals = None

                    # Apply rotation then translation only if validated
                    if rot_mat is not None:
                        try:
                            combined.apply_transform(rot_mat)
                        except Exception:
                            if args and getattr(args, "debug", False):
                                print(f"[DEBUG] Failed to apply rotation for {raw_key}; skipping rotation.")
                    if tvals is not None:
                        try:
                            combined.apply_translation(tvals)
                        except Exception:
                            if args and getattr(args, "debug", False):
                                print(f"[DEBUG] Failed to apply translation for {raw_key}; skipping translation.")

                # --- Post-transform robustness: normalize numeric ranges & recompute normals ---
                try:
                    # force float32 and limit decimals to reduce export noise
                    verts = np.asarray(combined.vertices, dtype=np.float32)
                    verts = np.round(verts, 6)
                    combined.vertices = verts

                    # remove degenerate / duplicate / unreferenced geometry
                    try:
                        combined.remove_degenerate_faces()
                        combined.remove_duplicate_faces()
                        combined.remove_unreferenced_vertices()
                    except Exception:
                        pass

                    # ensure there are faces and triangle topology
                    if not hasattr(combined, "faces") or combined.faces.size == 0:
                        # skip export of empty/invalid mesh
                        raise RuntimeError("mesh has no faces after cleanup")

                    # recompute normals so OBJ contains vn lines
                    try:
                        combined.fix_normals()
                    except Exception:
                        pass
                except Exception as ex:
                    # If severe problems, skip this model export and log
                    with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
                        log_file.write(f"SKIP EXPORT (invalid geometry) {raw_key}: {ex}\n")
                    if orp:
                        orp.write(f"{raw_key},{voxel_count},,ERROR_EXPORT_INVALID_GEOM,,,\n")
                    continue

                # Apply axes remap if requested (remap_R maps new = R @ old)
                if remap_R is not None:
                    R4 = np.eye(4, dtype=float)
                    R4[:3, :3] = remap_R
                    combined.apply_transform(R4)

                # --- Stable OBJ export ------------------------------
                export_ok = False
                try:
                    export_ok = apply_maya_compatibility_export(combined, target_path, debug=getattr(args, "debug", False), hard_edges=args.hard_edges)
                except Exception as ex:
                    if getattr(args, "debug", False):
                        print(f"[DEBUG] apply_maya_compatibility_export threw: {ex}")

                if not export_ok:
                    # Last-resort attempts: try stable writer on original, then trimesh default exporter
                    try:
                        write_stable_obj(target_path, combined)
                        export_ok = True
                    except Exception:
                        try:
                            tmp = target_path + ".tmp.obj"
                            combined.export(tmp, file_type='obj')
                            os.replace(tmp, target_path)
                            export_ok = True
                        except Exception as ex2:
                            export_ok = False
                            if getattr(args, "debug", False):
                                print(f"[DEBUG] Final fallback export failed for {target_path}: {ex2}")

                if not export_ok:
                    with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
                        log_file.write(f"ERROR: Failed to export {raw_key} as {os.path.basename(target_path)}\n")
                else:
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

    if orp:
        orp.close()

    try:
        with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
            log_file.write("\n--- Layer Name Mapping ---\n")
            for raw_key in ordered_parts:
                assigned = name_map.get(raw_key, raw_key)
                cnt = stats[raw_key]["count"]
                log_file.write(f"{raw_key:<12} {cnt:10d} {str(assigned):<30}\n")
            log_file.write(f"\nCompleted export of {vox_name} to {EXPORT_ROOT}\n")
    except Exception as ex:
        print(f"Failed to finalize export log: {ex}")

print("Export process completed.")