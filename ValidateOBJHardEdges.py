# ValidateOBJHardEdges.py
# Quick validator that ensures exported OBJs include vn entries and faces reference them.
import os
import sys
import glob
import re
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Validate exported OBJ files for vn references.")
parser.add_argument("--vox", help="Optional: limit to a single exported_meshes/<vox> folder (basename).")
parser.add_argument("--report", help="Path to write validation report (default reports/OBJValidationReport.txt).")
args = parser.parse_args()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_PATH = args.report or os.path.join(REPORTS_DIR, "OBJValidationReport.txt")

FACE_TOKEN_RE = re.compile(r'(-?\d+)(?:/(-?\d*)?(?:/(-?\d+))?)?$')

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

def validate_obj(path):
    problems = []
    vcount = 0
    vncount = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("v "):
            vcount += 1
        elif s.startswith("vn "):
            vncount += 1
    # require at least one vn
    if vncount == 0:
        problems.append("no_vn_lines")
    # check faces reference vn indices (if faces exist)
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
                # check vn index if present
                vn_tok = m.group(3)
                if vn_tok:
                    try:
                        ni = int(vn_tok)
                        if ni > 0 and ni > vncount:
                            problems.append(f"line_{i}: vn_index_{ni}_gt_vncount_{vncount}")
                    except Exception:
                        problems.append(f"line_{i}: bad_vn_token_{tok}")
                else:
                    # no vn referenced by face token; treat as problem for Maya compatibility
                    problems.append(f"line_{i}: face_token_has_no_vn_{tok}")
    return problems

def write_report(lines):
    tmp = REPORT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", errors="ignore") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, REPORT_PATH)

def main():
    objs = find_objs(args.vox)
    lines = []
    if not objs:
        lines.append("No .obj files found under exported_meshes (or specified vox).")
        write_report(lines)
        print(lines[-1])
        return 1
    problems_found = False
    for p in objs:
        prs = validate_obj(p)
        if prs:
            problems_found = True
            lines.append(f"{os.path.relpath(p, BASE_DIR)}: PROBLEMS: " + ";".join(prs))
        else:
            lines.append(f"{os.path.relpath(p, BASE_DIR)}: OK")
    write_report(lines)
    print("Validation report written to:", REPORT_PATH)
    return 2 if problems_found else 0

if __name__ == "__main__":
    sys.exit(main())