# -*- coding: utf-8 -*-
"""
ExportMeshes.py
Exports per-Part meshes from .vox using Vox200Parser, applies per-model transforms (if present),
and remaps axes for DCC. Final mesh scaling (--mesh-scale) is applied only after
voxel->mesh conversion and VOX transforms to preserve silhouette integrity.

Features:
 - CLI: --mesh-scale, --scale-factor (legacy), --ignore-model-scale, --no-extent-correction, --to-maya, --axes-map, --vox, --debug
 - Robust logging: exported_meshes/<vox>/ExportLog.txt and reports/ExportMeshesFatal_*.log on fatal errors.
 - Defensive: per-part exceptions are logged and processing continues.
"""
from __future__ import annotations

import os
import sys
import math
import traceback
import argparse
import datetime
import numpy as np
import trimesh
from trimesh.transformations import quaternion_matrix, euler_matrix, translation_matrix

from Vox200Parser import Vox200Parser

# CLI
cli = argparse.ArgumentParser(description="Export .vox parts as .obj with optional per-model transforms and axes remap.")
cli.add_argument("--to-maya", action="store_true", help="Remap exported geometry into Maya's world (preset axes map).")
cli.add_argument("--axes-map", help="Custom axes map, e.g. 'x,z,-y'.")
cli.add_argument("--vox", help="Optional: limit to a single .vox basename (no extension).")
cli.add_argument("--debug", action="store_true", help="Enable debug logging.")
cli.add_argument("--voxel-size", type=float, default=1.0, help="Size of one voxel in output units (default 1.0).")
cli.add_argument("--scale-factor", type=float, default=1.0, help="(Legacy) global scale multiplier - kept for backwards compat.")
cli.add_argument("--mesh-scale", type=float, default=1.0, help="Final uniform scale applied to mesh after conversion (default 1.0).")
cli.add_argument("--no-extent-correction", action="store_true", help="Skip extent-based ensure_voxel_scale step.")
cli.add_argument("--ignore-model-scale", action="store_true", help="Ignore per-model VOX 's' scale so voxels are not scaled prior to mesh export.")
args = cli.parse_args()

# Axis remap preset
AXES_MAP_SPEC = "x,z,-y" if args.to_maya else (args.axes_map or None)

# Decide final mesh scale:
EPS = 1e-12
if abs(args.mesh_scale - 1.0) > EPS:
    MESH_SCALE = float(args.mesh_scale)
elif abs(args.scale_factor - 1.0) > EPS:
    MESH_SCALE = float(args.scale_factor)
else:
    MESH_SCALE = 1.0

# ----------------- Helpers -----------------

def sanitize_filename(name: str) -> str:
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

def get_axes_matrix(axes_spec: str) -> np.ndarray:
    if not axes_spec:
        return np.identity(4)
    axes = axes_spec.split(',')
    if len(axes) != 3:
        return np.identity(4)
    T = np.zeros((4,4), dtype=float)
    T[3,3] = 1.0
    orig_map = {'x':0,'y':1,'z':2}
    for i, axis in enumerate(axes):
        is_neg = axis.startswith('-')
        orig_axis_name = axis.lstrip('-').strip()
        if orig_axis_name in orig_map:
            orig_index = orig_map[orig_axis_name]
            T[i, orig_index] = -1.0 if is_neg else 1.0
        else:
            if args.debug:
                print(f"[DEBUG] get_axes_matrix: unknown axis token '{axis}'")
    return T

def remap_axes(mesh: trimesh.Trimesh, axes_spec: str) -> trimesh.Trimesh:
    if not axes_spec:
        return mesh
    T = get_axes_matrix(axes_spec)
    mesh.apply_transform(T)
    return mesh

def _reorder_quaternion_for_trimesh(q):
    if q is None:
        return None
    try:
        qf = tuple(float(x) for x in q)
    except Exception:
        return None
    if len(qf) != 4:
        return None
    abs_vals = [abs(x) for x in qf]
    if abs_vals[3] >= max(abs_vals[0], abs_vals[1], abs_vals[2]):
        return (qf[0], qf[1], qf[2], qf[3])
    return (qf[1], qf[2], qf[3], qf[0])

def _safe_translation_vec(t):
    if t is None:
        return None
    try:
        tv = np.asarray(t, dtype=float)
    except Exception:
        return None
    if not np.isfinite(tv).all():
        return None
    max_abs = 1e5
    if np.any(np.abs(tv) > max_abs):
        tv = np.clip(tv, -max_abs, max_abs)
        if args.debug:
            print(f"[DEBUG] clamped translation to {tv}")
    return tv

def ensure_voxel_scale(mesh, positions, voxel_size=1.0, export_log_path=None, debug=False):
    try:
        if positions is None or positions.size == 0:
            if debug:
                print("[SCALE] no positions provided; skipping")
            return False
        positions = np.asarray(positions, dtype=float)
        pmin = positions.min(axis=0)
        pmax = positions.max(axis=0)
        expected_ext = ((pmax - pmin) + 1.0) * float(voxel_size)
        bounds = np.asarray(mesh.bounds, dtype=float)
        mesh_ext = bounds[1] - bounds[0]
        eps = 1e-8
        valid = mesh_ext > eps
        if not np.any(valid):
            if debug:
                print(f"[SCALE] invalid mesh extents: {mesh_ext}")
            return False
        scale_factors = np.empty(3, dtype=float)
        scale_factors.fill(np.nan)
        for i in range(3):
            if mesh_ext[i] > eps:
                scale_factors[i] = expected_ext[i] / mesh_ext[i]
        valid_sf = ~np.isnan(scale_factors)
        if not np.any(valid_sf):
            if debug:
                print("[SCALE] no valid scale factors computed; skipping")
            return False
        uni_scale = float(np.median(scale_factors[valid_sf]))
        if abs(uni_scale - 1.0) > 1e-6:
            try:
                mesh_centroid = np.asarray(mesh.centroid, dtype=float)
            except Exception:
                mesh_centroid = (bounds[0] + bounds[1]) / 2.0
            T1 = np.eye(4); T1[:3,3] = -mesh_centroid
            S = np.eye(4); S[:3,:3] *= uni_scale
            T2 = np.eye(4); T2[:3,3] = mesh_centroid
            M = T2.dot(S).dot(T1)
            mesh.apply_transform(M)
            msg = f"[SCALE] applied uniform scale {uni_scale:.6f} to match voxel extents (expected {expected_ext.tolist()}, pre-extents {mesh_ext.tolist()})"
            if debug:
                print(msg)
            if export_log_path:
                try:
                    with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                        lf.write(msg + "\n")
                except Exception:
                    pass
        else:
            if debug:
                print("[SCALE] mesh scale ~= 1.0; no action")
        return True
    except Exception as ex:
        if debug:
            print(f"[SCALE] failed: {ex}")
        return False

def verify_pre_scale_integrity(mesh, positions, export_log_path=None, tol=0.20):
    """
    Compare voxel-expected extents to mesh bounds before final mesh-scale.
    Returns True if mesh extents are within tol (relative) of expected extents.
    tol is the allowed fractional deviation (default 20%).
    This function logs details to export_log_path when provided.
    """
    try:
        if positions is None or positions.size == 0:
            return True
        positions = np.asarray(positions, dtype=float)
        pmin = positions.min(axis=0)
        pmax = positions.max(axis=0)
        expected_ext = ((pmax - pmin) + 1.0) * float(args.voxel_size)
        bounds = np.asarray(mesh.bounds, dtype=float)
        mesh_ext = bounds[1] - bounds[0]
        eps = 1e-8
        # Avoid division by zero: where expected_ext is tiny, skip that axis
        ratios = []
        for i in range(3):
            if expected_ext[i] <= eps or mesh_ext[i] <= eps:
                ratios.append(1.0)
            else:
                ratios.append(mesh_ext[i] / expected_ext[i])
        max_dev = max(abs(r - 1.0) for r in ratios)
        if export_log_path:
            try:
                with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                    lf.write(f"[INTEGRITY] expected_ext={expected_ext.tolist()} mesh_ext={mesh_ext.tolist()} ratios={ratios} max_dev={max_dev:.4f} tol={tol}\n")
            except Exception:
                pass
        return (max_dev <= tol)
    except Exception as ex:
        if export_log_path:
            try:
                with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                    lf.write(f"[INTEGRITY_ERROR] {ex}\n")
            except Exception:
                pass
        return True

def apply_maya_compatibility_export(mesh: trimesh.Trimesh, export_path: str) -> bool:
    # manual OBJ writer, returns False if mesh empty
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces') or len(getattr(mesh, 'faces', [])) == 0:
        return False
    try:
        mesh.process()
        try:
            if mesh.faces.ndim == 2 and mesh.faces.shape[1] != 3:
                mesh = mesh.copy().triangulate()
        except Exception:
            mesh = mesh.copy().triangulate()
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals.shape[0] != mesh.vertices.shape[0]:
            mesh.compute_vertex_normals()
        temp_path = export_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(f"# MagicaVoxel Export - Processed by ExportMeshes.py\n")
            f.write(f"o {os.path.splitext(os.path.basename(export_path))[0]}\n")
            for v in mesh.vertices:
                f.write('v {:.6f} {:.6f} {:.6f}\n'.format(*v))
            for vn in mesh.vertex_normals:
                f.write('vn {:.6f} {:.6f} {:.6f}\n'.format(*vn))
            for face in mesh.faces:
                idxs = (face + 1).tolist()
                if len(idxs) == 3:
                    v1, v2, v3 = idxs
                    f.write(f'f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n')
                else:
                    for a in range(1, len(idxs)-1):
                        f.write(f'f {idxs[0]}//{idxs[0]} {idxs[a]}//{idxs[a]} {idxs[a+1]}//{idxs[a+1]}\n')
        os.replace(temp_path, export_path)
        return True
    except Exception as ex:
        if args.debug:
            print(f"[DEBUG] Failed manual OBJ export for {os.path.basename(export_path)}: {ex}")
            traceback.print_exc(file=sys.stdout)
        return False

# ----------------- Main Export -----------------

def run_export(vox_file, AXES_MAP_SPEC):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    os.makedirs(EXPORT_ROOT, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    vox_path = os.path.join(BASE_DIR, vox_file)
    vox_name = os.path.splitext(vox_file)[0]
    sub_export_dir = os.path.join(EXPORT_ROOT, vox_name)
    os.makedirs(sub_export_dir, exist_ok=True)

    orp_path = os.path.join(REPORTS_DIR, f"VoxelOverrideReport_{vox_name}.txt")
    export_log_path = os.path.join(sub_export_dir, "ExportLog.txt")

    # create early ExportLog so failures are visible
    try:
        with open(export_log_path, "w", encoding="utf-8", errors="replace") as lf:
            lf.write(f"Export run for {vox_file} at {datetime.datetime.utcnow().isoformat()}Z\n")
            lf.write(f"mesh_scale={MESH_SCALE}  scale_factor={args.scale_factor}  ignore_model_scale={args.ignore_model_scale}\n")
    except Exception:
        pass

    try:
        parser = Vox200Parser(vox_path).parse()
    except Exception as ex:
        err = f"Failed to parse {vox_file}: {ex}"
        print(err)
        try:
            with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                lf.write(err + "\n")
                lf.write(traceback.format_exc() + "\n")
        except Exception:
            pass
        try:
            with open(orp_path, "w", encoding="utf-8", errors="replace") as orp:
                orp.write(f"FATAL_ERROR: Failed to parse VOX file: {ex}\n")
        except Exception:
            pass
        return

    voxels_by_layer = getattr(parser, 'voxels_by_layer', {})
    name_map = getattr(parser, 'layer_name_map', {})
    raw_map = getattr(parser, 'raw_part_name_map', None)
    model_transforms = getattr(parser, 'model_transforms', {})

    stats = {}
    for raw_key, voxels in voxels_by_layer.items():
        stats[raw_key] = {"count": len(voxels), "centroid": compute_centroid(voxels)}

    ordered_parts = sorted(voxels_by_layer.keys(), key=lambda k: int(k.split('_')[1]) if k.startswith('Part_') else 0)
    AXES_TRANSFORM = get_axes_matrix(AXES_MAP_SPEC) if AXES_MAP_SPEC else np.identity(4)
    used_filenames = set()

    # open override report file
    try:
        orp = open(orp_path, "w", encoding="utf-8", errors="replace")
        orp.write(f"Voxel Override Report for: {vox_file}\n")
        orp.write("RawPart,VoxelCount,Centroid (x,y,z),ParserAssigned,FinalName,NameSource,Reason\n")
    except Exception:
        orp = None

    integrity_fail_count = 0

    for raw_key in ordered_parts:
        voxels = voxels_by_layer.get(raw_key, [])
        voxel_count = len(voxels)
        if voxel_count == 0:
            continue

        parser_assigned = (raw_map.get(raw_key) if raw_map is not None else name_map.get(raw_key)) or ""
        final_name = parser_assigned or raw_key
        name_source = "layer_name_map"
        reason = ""

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
            positions = np.array([[v.x, v.y, v.z] for v in voxels], dtype=float)
            cubes = []
            for pos in positions:
                cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
                cube.apply_translation(pos + 0.5)
                cubes.append(cube)
            if not cubes:
                continue
            combined = trimesh.util.concatenate(cubes)

            # model transforms
            try:
                model_idx = int(raw_key.split('_', 1)[1])
            except Exception:
                model_idx = None
            model_transform_data = model_transforms.get(model_idx) or {}

            # diagnostic: log model-level 's' (if any) and whether it was applied
            s_val = model_transform_data.get('s')
            try:
                with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                    lf.write(f"[MODEL_TRANSFORM] {raw_key} model_idx={model_idx} model_s={s_val!r} ignore_model_scale={args.ignore_model_scale}\n")
            except Exception:
                pass

            S_vox = np.eye(4)
            # apply model s only if not ignored
            s_val = model_transform_data.get('s')
            if s_val is not None and not args.ignore_model_scale:
                try:
                    if isinstance(s_val, (int, float)):
                        S_vox[:3, :3] *= float(s_val)
                    else:
                        sv = list(map(float, s_val))
                        S_vox[0,0], S_vox[1,1], S_vox[2,2] = sv[0], sv[1] if len(sv)>1 else sv[0], sv[2] if len(sv)>2 else sv[0]
                except Exception:
                    if args.debug:
                        print(f"[DEBUG] invalid model scale for {raw_key}: {s_val}")
            else:
                if s_val is not None and args.ignore_model_scale and args.debug:
                    try:
                        with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                            lf.write(f"[INFO] Ignored model 's' for {raw_key}: {s_val}\n")
                    except Exception:
                        pass

            # rotation
            R_vox = np.eye(4)
            q = model_transform_data.get('r')
            if q is not None:
                q_safe = _reorder_quaternion_for_trimesh(q)
                if q_safe is not None:
                    try:
                        R_vox = quaternion_matrix(q_safe)
                    except Exception:
                        if args.debug:
                            print(f"[DEBUG] quaternion_matrix failed for {raw_key} q={q_safe}")
            else:
                e = model_transform_data.get('e')
                if e is not None:
                    try:
                        R_vox = euler_matrix(float(e[0]), float(e[1]), float(e[2]), axes='sxyz')
                    except Exception:
                        if args.debug:
                            print(f"[DEBUG] invalid euler for {raw_key}: {e}")

            # translation
            t = _safe_translation_vec(model_transform_data.get('t'))
            T_vox = translation_matrix(t) if t is not None else np.eye(4)

            # Compose and apply
            vox_transform = T_vox.dot(R_vox).dot(S_vox)
            combined.apply_transform(vox_transform)

            # log bounds after applying voxel transforms (before extent correction)
            try:
                bounds_after_vox = combined.bounds.copy()
                with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                    lf.write(f"[BOUNDS_AFTER_VOX_TRANSFORM] {raw_key} bounds={bounds_after_vox.tolist()}\n")
            except Exception:
                pass

            # axis remap
            if AXES_MAP_SPEC:
                combined = remap_axes(combined, AXES_MAP_SPEC)

            # extent correction
            try:
                if not args.no_extent_correction:
                    ensure_voxel_scale(combined, positions, voxel_size=args.voxel_size,
                                       export_log_path=os.path.join(REPORTS_DIR, f"ScaleFix_{safe_name}.log"),
                                       debug=args.debug)
            except Exception as ex:
                if args.debug:
                    print(f"[DEBUG] ensure_voxel_scale failed for {safe_name}: {ex}")
                try:
                    with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                        lf.write(f"[DEBUG] ensure_voxel_scale failed for {safe_name}: {ex}\n")
                except Exception:
                    pass

            # integrity check (logs to ExportLog.txt)
            pre_scale_ok = True
            try:
                pre_scale_ok = verify_pre_scale_integrity(combined, positions, export_log_path=export_log_path)
            except Exception:
                pre_scale_ok = True
            if not pre_scale_ok:
                integrity_fail_count += 1
            if not pre_scale_ok and args.debug:
                msg = f"[WARN] Pre-scale integrity check failed for {raw_key} - possible early scaling."
                print(msg)
                try:
                    with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                        lf.write(msg + "\n")
                except Exception:
                    pass

            # final mesh scale (applied to mesh only)
            if abs(MESH_SCALE - 1.0) > 1e-12:
                S_extra = np.eye(4); S_extra[:3,:3] *= float(MESH_SCALE)
                bounds_before = combined.bounds.copy() if hasattr(combined, "bounds") else None
                combined.apply_transform(S_extra)
                bounds_after = combined.bounds.copy() if hasattr(combined, "bounds") else None
                if args.debug:
                    try:
                        with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                            lf.write(f"[DEBUG] Part={raw_key} applied final mesh scale={MESH_SCALE}\n")
                            lf.write(f"[DEBUG]   bounds_before={bounds_before.tolist() if bounds_before is not None else None}\n")
                            lf.write(f"[DEBUG]   bounds_after ={bounds_after.tolist() if bounds_after is not None else None}\n")
                    except Exception:
                        pass

            # Ensure there is geometry to write; try repair/fallback if empty
            faces_count = getattr(combined, "faces", None)
            face_len = 0
            try:
                face_len = int(len(faces_count)) if faces_count is not None else 0
            except Exception:
                face_len = 0

            # If no faces, try to repair / create fallback
            fallback_used = False
            if face_len == 0:
                # try process, fill holes, recompute
                try:
                    combined.process()
                    if hasattr(combined, "faces"):
                        face_len = len(combined.faces)
                except Exception:
                    pass

            if face_len == 0:
                # attempt convex hull as fallback
                try:
                    hull = combined.convex_hull
                    if hull is not None and hasattr(hull, "faces") and len(hull.faces) > 0:
                        combined = hull
                        fallback_used = True
                        try:
                            with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                                lf.write(f"[FALLBACK] Used convex_hull for {raw_key}\n")
                        except Exception:
                            pass
                        face_len = len(getattr(combined, "faces", []))
                except Exception:
                    pass

            if face_len == 0:
                # ultimate fallback: write a tiny cube at centroid so you get an OBJ file
                try:
                    cen = np.array([0.0, 0.0, 0.0])
                    if positions is not None and positions.size:
                        cen = positions.mean(axis=0) + 0.5
                    tiny = trimesh.creation.box(extents=(0.001, 0.001, 0.001))
                    tiny.apply_translation(cen)
                    combined = tiny
                    fallback_used = True
                    try:
                        with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                            lf.write(f"[FALLBACK] Wrote tiny placeholder cube for {raw_key}\n")
                    except Exception:
                        pass
                except Exception:
                    pass

            # export
            exported_ok = False
            try:
                # prefer manual writer for Maya compatibility when possible
                if not apply_maya_compatibility_export(combined, target_path):
                    combined.export(target_path, file_type='obj')
                exported_ok = True
            except Exception as ex:
                exported_ok = False
                try:
                    with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                        lf.write(f"ERROR exporting {raw_key}: {ex}\n")
                        lf.write(traceback.format_exc() + "\n")
                except Exception:
                    pass
                if args.debug:
                    print(f"[ERROR] failed to write OBJ for {raw_key}: {ex}")
                    traceback.print_exc()

            # log
            try:
                with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
                    if exported_ok:
                        log_file.write(f"Exported {raw_key} ({voxel_count} voxels) as {os.path.basename(target_path)}\n")
                    else:
                        log_file.write(f"FAILED_EXPORT {raw_key} ({voxel_count}) -> {os.path.basename(target_path)}\n")
            except Exception:
                pass

            try:
                if orp:
                    cent = stats[raw_key]["centroid"] if raw_key in stats else None
                    cent_str = ",".join([f"{x:.3f}" for x in cent]) if cent is not None else ""
                    orp.write(f"{raw_key},{voxel_count},{cent_str},{parser_assigned},{final_name},{name_source},{reason}\n")
            except Exception:
                pass

        except Exception as ex:
            # per-part failure: log and continue
            try:
                with open(export_log_path, "a", encoding="utf-8", errors="replace") as lf:
                    lf.write(f"EXCEPTION exporting {raw_key}: {ex}\n")
                    lf.write(traceback.format_exc() + "\n")
            except Exception:
                pass
            if args.debug:
                print(f"[EXCEPTION] exporting {raw_key}: {ex}")
                traceback.print_exc()
            try:
                if orp:
                    orp.write(f"{raw_key},{voxel_count},,ERROR_EXPORT,,,{ex}\n")
            except Exception:
                pass

    # final summary
    try:
        with open(export_log_path, "a", encoding="utf-8", errors="replace") as log_file:
            log_file.write("\n--- Layer Name Mapping ---\n")
            for raw_key in ordered_parts:
                assigned = name_map.get(raw_key, raw_key)
                cnt = stats[raw_key]["count"]
                log_file.write(f"  {raw_key:<10} ({cnt:4d} voxels) => {assigned}\n")
    except Exception:
        pass

    try:
        if orp:
            orp.close()
    except Exception:
        pass

# Entrypoint
if __name__ == "__main__":
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(BASE_DIR)

        vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]
        if args.vox:
            target = f"{args.vox}.vox"
            if target in vox_files:
                vox_files = [target]
            else:
                print(f"Specified vox not found: {args.vox}")
                sys.exit(1)
        if not vox_files:
            print("No .vox files found.")
            sys.exit(0)

        for vox_file in vox_files:
            try:
                run_export(vox_file, AXES_MAP_SPEC)
            except Exception as ex_main:
                # fatal per-file: write report
                try:
                    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
                    os.makedirs(reports_dir, exist_ok=True)
                    fatal_path = os.path.join(reports_dir, f"ExportMeshesFatal_{os.path.splitext(vox_file)[0]}.log")
                    with open(fatal_path, "w", encoding="utf-8", errors="replace") as fh:
                        fh.write(f"FATAL exception exporting {vox_file}: {ex_main}\n")
                        fh.write(traceback.format_exc())
                except Exception:
                    pass
                print(f"FATAL ERROR in exporting {vox_file}: {ex_main}")
                if args.debug:
                    traceback.print_exc()

    except Exception as e:
        print(f"FATAL ERROR in ExportMeshes.py main loop: {e}")
        if args.debug:
            traceback.print_exc()
        try:
            reports_dir = os.path.join(os.path.dirname(__file__), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            fatal_path = os.path.join(reports_dir, f"ExportMeshesFatal_main.log")
            with open(fatal_path, "w", encoding="utf-8", errors="replace") as fh:
                fh.write(f"FATAL main exception: {e}\n")
                fh.write(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)