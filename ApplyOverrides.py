import os
import csv
import json
import shutil
import argparse
import datetime
import sys

BASE_DIR = os.path.dirname(__file__)
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Apply explicit Part_X -> name overrides to exported .obj files.")
parser.add_argument("--vox", required=True, help="vox basename (e.g. Character)")
parser.add_argument("--overrides", default="name_overrides.json", help="JSON file with {\"Part_32\":\"Head\",...}")
parser.add_argument("--commit", action="store_true", help="Perform renames. Without --commit runs dry-run.")
args = parser.parse_args()

vox = args.vox
vox_base = vox
final_csv = os.path.join(REPORTS_DIR, f"FinalMapping_{vox_base}.csv")

# If final mapping CSV missing, write an error manifest and exit with non-zero
manifest_path = os.path.join(REPORTS_DIR, f"ApplyOverridesManifest_{vox_base}.json")
if not os.path.exists(final_csv):
    err_manifest = {
        "vox": vox + ".vox",
        "timestamp": datetime.datetime.now().isoformat(),
        "error": f"FinalMapping CSV not found: {final_csv}",
        "renames": []
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(err_manifest, mf, indent=2)
    print(f"Final mapping CSV not found: {final_csv}")
    print(f"Wrote error manifest: {manifest_path}")
    raise SystemExit(1)

# Resolve overrides path: prefer explicit path, else look in reports/
overrides_arg = args.overrides or ""
overrides_path = overrides_arg
if not os.path.isabs(overrides_path):
    # if file exists relative to BASE_DIR use that, else try reports/
    if not os.path.exists(overrides_path):
        candidate = os.path.join(REPORTS_DIR, overrides_path)
        if os.path.exists(candidate):
            overrides_path = candidate
        else:
            # also try basename in reports (e.g. name_overrides_highconf_Character.json)
            candidate2 = os.path.join(REPORTS_DIR, os.path.basename(overrides_path))
            if os.path.exists(candidate2):
                overrides_path = candidate2

if not os.path.exists(overrides_path):
    err_manifest = {
        "vox": vox + ".vox",
        "timestamp": datetime.datetime.now().isoformat(),
        "error": f"Overrides file not found: {args.overrides} (tried {overrides_path})",
        "renames": []
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(err_manifest, mf, indent=2)
    print(f"Overrides file not found: {args.overrides}")
    print(f"Wrote error manifest: {manifest_path}")
    raise SystemExit(1)

with open(overrides_path, "r", encoding="utf-8") as f:
    overrides = json.load(f)

# read final mapping CSV to map raw_part -> export_filename
part_to_file = {}
with open(final_csv, newline="", encoding="utf-8") as c:
    reader = csv.DictReader(c)
    for row in reader:
        # expected headers: ExportFile,RawPart,IntendedName,VoxelCount,CentroidDistance
        raw = (row.get("RawPart") or row.get(" RawPart") or row.get("RawPart ") or "").strip()
        export_file = (row.get("ExportFile") or row.get(" ExportFile") or "").strip()
        if raw and export_file:
            part_to_file[raw] = export_file

exported_dir = os.path.join(EXPORT_ROOT, vox_base)
if not os.path.isdir(exported_dir):
    err_manifest = {
        "vox": vox + ".vox",
        "timestamp": datetime.datetime.now().isoformat(),
        "error": f"Export folder not found: {exported_dir}",
        "renames": []
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(err_manifest, mf, indent=2)
    print(f"Export folder not found: {exported_dir}")
    print(f"Wrote error manifest: {manifest_path}")
    raise SystemExit(1)

# Build plan: for each override entry find current file and desired target path
plan = []  # (src_path, target_path, backup_path, part, src_name, tgt_name)
used_targets = set()
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
for part, desired in overrides.items():
    src_name = part_to_file.get(part)
    if not src_name:
        print(f"Warning: no exported file mapped to {part} (check {final_csv})")
        continue
    src_path = os.path.join(exported_dir, src_name)
    if not os.path.exists(src_path):
        print(f"Warning: source file not found: {src_path}")
        continue
    safe_name = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in (desired or "")).strip()
    if not safe_name:
        safe_name = part
    target_name = f"{safe_name}.obj"
    target_path = os.path.join(exported_dir, target_name)
    # If two overrides target same name, append numeric suffix
    base, ext = os.path.splitext(target_name)
    idx = 1
    while target_path in used_targets or (os.path.exists(target_path) and os.path.abspath(target_path) == os.path.abspath(src_path) and target_path in used_targets):
        alt = f"{base}_{idx}{ext}"
        target_path = os.path.join(exported_dir, alt)
        idx += 1
    used_targets.add(target_path)
    backup = None
    backup_path = None
    if os.path.exists(target_path) and os.path.abspath(target_path) != os.path.abspath(src_path):
        backup = f"{os.path.basename(target_path)}.bak.{timestamp}"
        backup_path = os.path.join(exported_dir, backup)
    plan.append((src_path, target_path, backup_path, part, src_name, os.path.basename(target_path)))

# Write dry-run report and manifest (always write manifest so Undo can inspect)
report_path = os.path.join(REPORTS_DIR, f"ApplyOverridesReport_{vox_base}.txt")
manifest = {"vox": vox + ".vox", "timestamp": datetime.datetime.now().isoformat(), "renames": []}
with open(report_path, "w", encoding="utf-8") as rep:
    rep.write(f"ApplyOverrides Report for {vox_base}  (commit={args.commit})\n\n")
    if not plan:
        rep.write("No overrides planned.\n")
    else:
        rep.write("Planned renames:\n")
        for src, tgt, bkp, part, src_name, tgt_name in plan:
            rep.write(f"  {os.path.basename(src)} -> {os.path.basename(tgt)}   (raw {part} => '{tgt_name}')")
            if bkp:
                rep.write(f"  [will backup existing {os.path.basename(tgt)} -> {os.path.basename(bkp)}]")
            rep.write("\n")

for src, tgt, bkp, part, src_name, tgt_name in plan:
    manifest["renames"].append({"src": os.path.basename(src), "dst": os.path.basename(tgt), "backup": os.path.basename(bkp) if bkp else None, "part": part})

with open(manifest_path, "w", encoding="utf-8") as mf:
    json.dump(manifest, mf, indent=2)

print(f"Report written: {report_path}")
print(f"Manifest written: {manifest_path}")

if not args.commit:
    print("Dry-run only. Re-run with --commit to perform renames.")
    raise SystemExit(0)

# commit: two-phase safe rename
temp_records = []
try:
    for src, tgt, bkp, part, src_name, tgt_name in plan:
        # move src -> temp
        tmp = os.path.join(exported_dir, f".tmp_override_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{os.path.basename(src)}")
        shutil.move(src, tmp)
        temp_records.append((tmp, tgt, bkp, src_name))
    # apply backups then move temp -> final
    for tmp, final_tgt, bkp, orig_src_name in temp_records:
        if os.path.exists(final_tgt):
            if bkp:
                shutil.move(final_tgt, os.path.join(exported_dir, bkp))
        shutil.move(tmp, final_tgt)
    # update manifest with actual backups (already recorded in manifest as planned)
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print("Renames applied. Backups created where needed.")
except Exception as ex:
    print("ERROR applying renames:", ex)
    # best-effort rollback: try to move any temps back to original names (not exhaustive)
    for tmp, final_tgt, bkp, orig_src_name in temp_records:
        if os.path.exists(tmp):
            dst = os.path.join(exported_dir, orig_src_name)
            shutil.move(tmp, dst)
    raise
