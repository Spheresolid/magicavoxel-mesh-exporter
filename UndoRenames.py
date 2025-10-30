import os
import json
import shutil
import argparse

BASE_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")

parser = argparse.ArgumentParser(description="Undo renames recorded in RenamingManifest_<vox>.json")
parser.add_argument("--vox", help="vox basename (e.g. Character) — required", required=True)
parser.add_argument("--commit", action="store_true", help="Perform undo. Without --commit, runs in preview mode.")
args = parser.parse_args()

manifest_path = os.path.join(REPORTS_DIR, f"RenamingManifest_{args.vox}.json")
if not os.path.exists(manifest_path):
    print(f"Manifest not found: {manifest_path}")
    raise SystemExit(1)

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

vox_name = os.path.splitext(manifest.get("vox", args.vox + ".vox"))[0] if manifest.get("vox") else args.vox
exported_dir = os.path.join(EXPORT_ROOT, vox_name)
if not os.path.isdir(exported_dir):
    print(f"Export folder not found: {exported_dir}")
    raise SystemExit(1)

undo_ops = []
for entry in manifest.get("renames", []):
    src = entry.get("src")
    dst = entry.get("dst")
    backup = entry.get("backup")  # may be None
    src_path = os.path.join(exported_dir, src)
    dst_path = os.path.join(exported_dir, dst)
    backup_path = os.path.join(exported_dir, backup) if backup else None

    # If a backup exists, restore it to dst_path
    if backup and os.path.exists(backup_path):
        undo_ops.append(("restore_backup", backup_path, dst_path))
    else:
        # If dst exists and src missing, move dst back to src
        if os.path.exists(dst_path) and not os.path.exists(src_path):
            undo_ops.append(("move_back", dst_path, src_path))
        else:
            undo_ops.append(("skip", src_path, dst_path))

# Preview
print(f"Manifest: {manifest_path}")
print(f"Export folder: {exported_dir}\n")
print("Planned undo operations:")
for op, a, b in undo_ops:
    if op == "restore_backup":
        print(f"  RESTORE: {os.path.basename(a)} -> {os.path.basename(b)}")
    elif op == "move_back":
        print(f"  MOVE BACK: {os.path.basename(a)} -> {os.path.basename(b)}")
    else:
        print(f"  SKIP (manual): src={os.path.basename(a)} dst={os.path.basename(b)}")

if not args.commit:
    print("\nDry-run. Rerun with --commit to perform the undo.")
    raise SystemExit(0)

# Perform undo
errors = []
for op, a, b in undo_ops:
    try:
        if op == "restore_backup":
            # move backup -> target (overwrite current if exists)
            if os.path.exists(b):
                tmp = b + ".preundo.bak"
                shutil.move(b, tmp)
            shutil.move(a, b)
            print(f"Restored backup: {os.path.basename(a)} -> {os.path.basename(b)}")
        elif op == "move_back":
            shutil.move(a, b)
            print(f"Moved back: {os.path.basename(a)} -> {os.path.basename(b)}")
        else:
            print(f"Skipped (manual): {os.path.basename(a)} -> {os.path.basename(b)}")
    except Exception as ex:
        errors.append((op, a, b, str(ex)))
        print(f"ERROR on {op} {a} -> {b}: {ex}")

if errors:
    print("\nCompleted with errors:")
    for e in errors:
        print(e)
else:
    print("\nUndo completed successfully.")