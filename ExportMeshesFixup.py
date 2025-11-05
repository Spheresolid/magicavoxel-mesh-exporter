# ExportMeshesFixup.py
# -*- coding: utf-8 -*-
"""
ExportMeshesFixup.py
Validate, sanitize, (optionally) hard-edge rewrite exported OBJ files so they import cleanly into Maya.

Behavior:
 - Always create reports/OBJValidationReport.txt header early.
 - Load each OBJ (trimesh.load). If load fails, attempt text salvage.
 - Do repairs, normals, recentering, precision reductions.
 - Prefer hard-edge rewrite when --to-maya or --hard-edges is passed:
     write_hard_obj() duplicates vertices per face and writes per-face vn (flat shading).
 - After rewrite, validate that the OBJ contains vn lines and faces reference vn indices.
 - Write reports and return:
     0 = all OK,
     1 = no OBJ files found,
     2 = problems found/fixed or unrecoverable issues.
"""
import os
import sys
import argparse
import datetime

# Early header so a report file exists even if heavy imports fail
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
DEFAULT_REPORT_PATH = os.path.join(REPORTS_DIR, "OBJValidationReport.txt")
RUNLOG_PATH = os.path.join(REPORTS_DIR, "RunMagicaExport.log")
try:
    header = [
        f"ExportMeshesFixup init at {datetime.datetime.now().isoformat()}",
        f"base_dir: {BASE_DIR}",
        ""
    ]
    tmp_init = DEFAULT_REPORT_PATH + ".init"
    with open(tmp_init, "w", encoding="utf-8", errors="ignore") as hf:
        hf.write("\n".join(header) + "\n")
    os.replace(tmp_init, DEFAULT_REPORT_PATH)
    tmp_l = RUNLOG_PATH + ".init"
    with open(tmp_l, "w", encoding="utf-8", errors="ignore") as lf:
        lf.write("\n".join(header) + "\n")
    os.replace(tmp_l, RUNLOG_PATH)
except Exception:
    pass

# Heavy imports
import glob
import traceback
import re
import numpy as np
import trimesh

FACE_TOKEN_RE = re.compile(r'(-?\d+)(?:/(-?\d*)?(?:/(-?\d+))?)?$')

parser = argparse.ArgumentParser(description="Validate and (safely) fix exported OBJ files for Maya.")
parser.add_argument("--vox", help="Optional: limit to a single exported_meshes/<vox> folder (basename).")
parser.add_argument("--max-abs", type=float, default=1e5, help="Max allowed absolute coordinate before recentring (default 1e5).")
parser.add_argument("--report", help="Path to write validation report (default reports/OBJValidationReport.txt).")
parser.add_argument("--to-maya", action="store_true", help="Informational flag - enables hard-edge behavior.")
parser.add_argument("--hard-edges", action="store_true", help="Force hard-edged rewrite (duplicate vertices per face + vn per face).")
args = parser.parse_args()

REPORT_PATH = args.report or DEFAULT_REPORT_PATH
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
HARD_EDGES = bool(args.hard_edges) or bool(args.to_maya)

def find_objs(vox_filter=None):
    out = []
    base_dirs = []
    if vox_filter:
        d = os.path.join(EXPORT_ROOT, vox_filter)
        if os.path.isdir(d):
            base_dirs.append(d)
    else:
        for d in glob.glob(os.path.join(EXPORT_ROOT, "*")):
            if os.path.isdir(d):
                base_dirs.append(d)
    for bd in base_dirs:
        for p in glob.glob(os.path.join(bd, "*.obj")):
            out.append(p)
    return sorted(out)

def write_stable_obj(path, mesh):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
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
        f.write("# Stable OBJ exported by ExportMeshesFixup.py\n")
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
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=int)
    fmt = "{:.6f} {:.6f} {:.6f}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Hard-edged OBJ exported by ExportMeshesFixup.py\n")
        f.write("s off\n")
        idx = 1
        for face in faces:
            a = int(face[0]); b = int(face[1]); c = int(face[2])
            if a < 0 or b < 0 or c < 0:
                raise RuntimeError("Negative index in write_hard_obj")
            v0 = verts[a]; v1 = verts[b]; v2 = verts[c]
            e1 = v1 - v0; e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if not np.isfinite(norm) or norm == 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                n = n / norm
            f.write("v " + fmt.format(float(v0[0]), float(v0[1]), float(v0[2])))
            f.write("v " + fmt.format(float(v1[0]), float(v1[1]), float(v1[2])))
            f.write("v " + fmt.format(float(v2[0]), float(v2[1]), float(v2[2])))
            f.write("vn " + fmt.format(float(n[0]), float(n[1]), float(n[2])))
            f.write("vn " + fmt.format(float(n[0]), float(n[1]), float(n[2])))
            f.write("vn " + fmt.format(float(n[0]), float(n[1]), float(n[2])))
            f.write(f"f {idx}//{idx} {idx+1}//{idx+1} {idx+2}//{idx+2}\n")
            idx += 3

def validate_obj_file(path):
    """Return list of problems (empty means OK)."""
    problems = []
    vcount = 0; vncount = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for ln in lines:
        s = ln.strip()
        if s.startswith("v "):
            vcount += 1
        elif s.startswith("vn "):
            vncount += 1
    if vncount == 0:
        problems.append("no_vn_lines")
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("f "):
            toks = s[2:].split()
            if len(toks) < 3:
                problems.append(f"line_{i}: face_less_than_3")
                continue
            for tok in toks:
                m = FACE_TOKEN_RE.match(tok)
                if not m:
                    problems.append(f"line_{i}: malformed_face_token_{tok}")
                    continue
                vn_tok = m.group(3)
                if not vn_tok:
                    problems.append(f"line_{i}: face_token_has_no_vn_{tok}")
                else:
                    try:
                        ni = int(vn_tok)
                        if ni > vncount:
                            problems.append(f"line_{i}: vn_index_{ni}_gt_vncount_{vncount}")
                    except Exception:
                        problems.append(f"line_{i}: bad_vn_token_{tok}")
    return problems

def parse_obj_text_salvage(path):
    verts = []
    faces = []
    any_face = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("v "):
                parts = s.split()
                if len(parts) < 4:
                    continue
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except Exception:
                    continue
                verts.append([x, y, z])
            elif s.startswith("f "):
                any_face = True
                toks = s[2:].split()
                idxs = []
                for tok in toks:
                    m = FACE_TOKEN_RE.match(tok)
                    if not m:
                        continue
                    try:
                        vi = int(m.group(1))
                    except Exception:
                        continue
                    idxs.append(vi)
                if len(idxs) < 3:
                    continue
                for i in range(1, len(idxs)-1):
                    faces.append([idxs[0], idxs[i], idxs[i+1]])
    if not verts or not any_face or not faces:
        raise RuntimeError("No salvageable geometry")
    verts_a = np.array(verts, dtype=float)
    vcount = verts_a.shape[0]
    out_faces = []
    for fidx in faces:
        tri = []
        for idx in fidx:
            if idx < 0:
                pos = vcount + idx
            else:
                pos = idx - 1
            if pos < 0 or pos >= vcount:
                tri = None; break
            tri.append(pos)
        if tri is None:
            continue
        out_faces.append(tri)
    if not out_faces:
        raise RuntimeError("No valid faces after salvage")
    return verts_a, np.array(out_faces, dtype=int)

def safe_fix_mesh(path, max_abs):
    info = {"path": path, "fixed": False, "errors": []}
    mesh = None
    load_exc = None
    try:
        mesh = trimesh.load(path, force='mesh')
    except Exception as ex:
        load_exc = ex; mesh = None
    if mesh is None:
        try:
            verts, faces = parse_obj_text_salvage(path)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            info["fixed"] = True
        except Exception as ex:
            info["errors"].append(f"salvage_failed: {ex} (load_exc: {load_exc})")
            return info
    try:
        verts = np.asarray(mesh.vertices, dtype=float)
    except Exception as ex:
        info["errors"].append(f"vertex_access_failed: {ex}")
        return info

    # Fix non-finite vertices
    if verts.size and (not np.isfinite(verts).all()):
        info["fixed"] = True
        finite_mask = np.isfinite(verts).all(axis=1)
        finite_verts = verts[finite_mask] if finite_mask.any() else None
        centroid = finite_verts.mean(axis=0) if finite_verts is not None and finite_verts.size else np.array([0.0,0.0,0.0], dtype=float)
        for i in range(verts.shape[0]):
            if not np.isfinite(verts[i]).all():
                verts[i] = centroid
        mesh.vertices = verts

    # remove degenerate/duplicate/unreferenced
    try:
        before_faces = int(mesh.faces.shape[0]) if hasattr(mesh, "faces") else 0
        try: mesh.remove_degenerate_faces()
        except Exception: pass
        try: mesh.remove_duplicate_faces()
        except Exception: pass
        try: mesh.remove_unreferenced_vertices()
        except Exception: pass
        after_faces = int(mesh.faces.shape[0]) if hasattr(mesh, "faces") else 0
        if after_faces != before_faces:
            info["fixed"] = True
    except Exception:
        pass

    faces_count = int(mesh.faces.shape[0]) if hasattr(mesh, "faces") else 0
    if faces_count == 0:
        info["errors"].append("zero_faces_after_cleanup"); return info

    # recenter if coordinates huge
    try:
        bbox = np.array(mesh.bounds) if hasattr(mesh, "bounds") else None
    except Exception:
        bbox = None
    if bbox is not None:
        max_abs_coord = max(abs(bbox[0]).max(), abs(bbox[1]).max())
        if max_abs_coord > max_abs:
            info["fixed"] = True
            centroid = mesh.centroid if hasattr(mesh, "centroid") else np.mean(mesh.vertices, axis=0)
            try: mesh.apply_translation(-centroid)
            except Exception: pass

    # normals
    try:
        vn_ok = False
        try:
            if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None and np.isfinite(mesh.vertex_normals).all():
                vn_ok = True
        except Exception:
            vn_ok = False
        if not vn_ok:
            try:
                mesh.fix_normals()
                info["fixed"] = True
            except Exception:
                pass
    except Exception:
        pass

    # precision reduce
    try:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        verts = np.round(verts, 6)
        mesh.vertices = verts
    except Exception:
        pass

    # export (prefer hard edges when requested)
    try:
        tmp = path + ".clean.tmp.obj"
        if HARD_EDGES:
            write_hard_obj(tmp, mesh)
        else:
            write_stable_obj(tmp, mesh)
        # strip mtllib/usemtl
        with open(tmp, "r", encoding="utf-8", errors="ignore") as f_in:
            lines = [L for L in f_in.readlines() if not (L.strip().lower().startswith("mtllib") or L.strip().lower().startswith("usemtl"))]
        with open(path, "w", encoding="utf-8", errors="ignore") as f_out:
            f_out.writelines(lines)
        try: os.remove(tmp)
        except Exception: pass
    except Exception as ex:
        info["errors"].append(f"export_failed: {ex}"); return info

    # validation: ensure vn exist and faces reference vn
    try:
        probs = validate_obj_file(path)
        if probs:
            info["errors"].append("validation_failed:" + ";".join(probs))
            # if we didn't try hard edges yet, try now
            if not HARD_EDGES:
                try:
                    tmp2 = path + ".hard.tmp.obj"
                    write_hard_obj(tmp2, mesh)
                    with open(tmp2, "r", encoding="utf-8", errors="ignore") as f_in:
                        lines2 = [L for L in f_in.readlines() if not (L.strip().lower().startswith("mtllib") or L.strip().lower().startswith("usemtl"))]
                    with open(path, "w", encoding="utf-8", errors="ignore") as f_out:
                        f_out.writelines(lines2)
                    try: os.remove(tmp2)
                    except Exception: pass
                    info["fixed"] = True
                    # revalidate
                    reprob = validate_obj_file(path)
                    if reprob:
                        info["errors"].append("post_hard_rewrite_failed:" + ";".join(reprob))
                    else:
                        # success after hard rewrite
                        pass
                except Exception as ex:
                    info["errors"].append(f"hard_rewrite_failed:{ex}")
    except Exception as ex:
        info["errors"].append(f"validation_exception:{ex}")

    return info

def _write_reports_atomic(report_lines):
    try:
        tmp_r = REPORT_PATH + ".tmp"
        with open(tmp_r, "w", encoding="utf-8", errors="ignore") as rf:
            rf.write("\n".join(report_lines)); rf.write("\n")
        os.replace(tmp_r, REPORT_PATH)
    except Exception:
        try:
            with open(REPORT_PATH, "w", encoding="utf-8", errors="ignore") as rf:
                rf.write("\n".join(report_lines)); rf.write("\n")
        except Exception:
            pass
    try:
        tmp_l = RUNLOG_PATH + ".tmp"
        with open(tmp_l, "w", encoding="utf-8", errors="ignore") as lf:
            lf.write("\n".join(report_lines)); lf.write("\n")
        os.replace(tmp_l, RUNLOG_PATH)
    except Exception:
        try:
            with open(RUNLOG_PATH, "w", encoding="utf-8", errors="ignore") as lf:
                lf.write("\n".join(report_lines)); lf.write("\n")
        except Exception:
            pass

def main():
    report_lines = []
    problems = []
    try:
        objs = find_objs(args.vox)
        if not objs:
            report_lines.append("No .obj files found under exported_meshes (or specified vox).")
            _write_reports_atomic(report_lines); return 1
        for p in objs:
            try:
                info = safe_fix_mesh(p, args.max_abs)
                line = f"{os.path.relpath(p, BASE_DIR)}: fixed={info.get('fixed',False)}"
                if info.get("errors"):
                    line += " ERR=" + ";".join(info["errors"])
                    problems.append(info)
                report_lines.append(line)
            except Exception as ex:
                tb = traceback.format_exc()
                report_lines.append(f"{os.path.relpath(p, BASE_DIR)}: EXCEPTION {ex}")
                problems.append({"path": p, "errors": [str(ex), tb]})
        _write_reports_atomic(report_lines)
        print("OBJ validation/fixup report written to:", REPORT_PATH)
        if problems:
            print("Problems found/fixed for some OBJs. Inspect the report and the OBJ files.")
            return 2
        return 0
    except Exception as ex_main:
        tb_main = traceback.format_exc()
        report_lines.append(f"FATAL_EXCEPTION: {ex_main}"); report_lines.append(tb_main)
        _write_reports_atomic(report_lines)
        print("FATAL: ExportMeshesFixup encountered an exception. See", REPORT_PATH)
        return 2

if __name__ == "__main__":
    sys.exit(main())