import os
import glob
import sys
import struct
import re
from collections import namedtuple

Voxel = namedtuple("Voxel", ["x", "y", "z", "color_index"])

def sanitize_filename(name):
    return ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in (name or "")).strip()

def parse_vox_layers_and_models(filepath):
    layer_id_to_name = {}
    model_voxels = {}
    model_voxel_counts = {}
    layer_ids_in_order = []

    with open(filepath, "rb") as f:
        if f.read(4) != b"VOX ":
            raise ValueError("Not a valid VOX file")
        version = struct.unpack("<I", f.read(4))[0]

        model_counter = 0
        while True:
            chunk_header = f.read(12)
            if not chunk_header or len(chunk_header) < 12:
                break
            chunk_id, content_size, children_size = struct.unpack("<4sII", chunk_header)
            chunk_id = chunk_id.strip()
            content = f.read(content_size)

            if chunk_id == b"LAYR":
                layer_id = struct.unpack("<I", content[:4])[0]
                offset = 4
                # mirrors ExportMeshes.py: read attr_dict_len at offset 4
                if offset + 4 <= len(content):
                    attr_dict_len = struct.unpack("<I", content[offset:offset+4])[0]
                    offset += 4
                else:
                    attr_dict_len = 0
                name = None
                for _ in range(attr_dict_len):
                    if offset + 4 > len(content): break
                    key_len = struct.unpack("<I", content[offset:offset+4])[0]; offset += 4
                    if offset + key_len > len(content): break
                    key = content[offset:offset+key_len].decode("utf-8", errors="ignore"); offset += key_len
                    if offset + 4 > len(content): break
                    val_len = struct.unpack("<I", content[offset:offset+4])[0]; offset += 4
                    if offset + val_len > len(content): break
                    val = content[offset:offset+val_len].decode("utf-8", errors="ignore"); offset += val_len
                    if key == "_name":
                        name = val
                if name is not None:
                    layer_id_to_name[layer_id] = name
                    layer_ids_in_order.append(layer_id)

            elif chunk_id == b"XYZI":
                if len(content) < 4:
                    continue
                num_voxels = struct.unpack("<I", content[:4])[0]
                voxel_data = content[4:]
                voxels = []
                for i in range(num_voxels):
                    if i * 4 + 4 > len(voxel_data):
                        break
                    x, y, z, color_index = struct.unpack("BBBB", voxel_data[i * 4:(i + 1) * 4])
                    voxels.append(Voxel(x, y, z, color_index))
                model_voxels[model_counter] = voxels
                model_voxel_counts[model_counter] = num_voxels
                model_counter += 1

    # Build ordered list of non-"00" layer names (same logic as ExportMeshes.py)
    ordered_layer_names = []
    ordered_layer_ids = []
    for lid in layer_ids_in_order:
        lname = layer_id_to_name.get(lid, f"Layer_{lid}")
        if lname != "00":
            ordered_layer_names.append(lname)
            ordered_layer_ids.append(lid)

    return model_voxels, ordered_layer_names, model_voxel_counts

def read_exportlog_mappings(export_log_path):
    """
    Parse ExportLog.txt lines like:
      Exported: <fullpath> as <AssignedName> with <N> voxels
    returns dict filename -> assigned_name (from exporter)
    """
    mapping = {}
    if not os.path.exists(export_log_path):
        return mapping
    pattern = re.compile(r"Exported:\s*(.+?)\s+as\s+(.+?)\s+with\s+(\d+)\s+voxels", re.IGNORECASE)
    with open(export_log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                fullpath = m.group(1).strip()
                assigned = m.group(2).strip()
                filename = os.path.basename(fullpath)
                mapping[filename] = assigned
    return mapping

BASE_DIR = os.path.dirname(__file__)
os.chdir(BASE_DIR)

EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]
if not vox_files:
    print("No .vox files found.")
    sys.exit(0)

for vox_file in vox_files:
    print(f"Checking: {vox_file}")
    vox_path = os.path.join(BASE_DIR, vox_file)
    vox_base = os.path.splitext(vox_file)[0]
    model_voxels, ordered_layer_names, model_voxel_counts = parse_vox_layers_and_models(vox_path)

    # expected filenames computed with the exact same logic ExportMeshes.py uses
    expected_files = []
    idx = 0
    for model_id in sorted(model_voxels.keys()):
        voxels = model_voxels[model_id]
        if len(voxels) == 0:
            # ExportMeshes.py skips empty models
            continue
        if idx < len(ordered_layer_names):
            name = ordered_layer_names[idx]
        else:
            name = f"Part_{model_id}"
        safe_name = sanitize_filename(name)
        expected_files.append((model_id, name, f"{safe_name}.obj", len(voxels)))
        idx += 1

    exported_dir = os.path.join(EXPORT_ROOT, vox_base)
    exported_files = set(os.path.basename(p) for p in glob.glob(os.path.join(exported_dir, "*.obj"))) if os.path.isdir(exported_dir) else set()

    expected_set = set(e[2] for e in expected_files)
    missing = sorted(expected_set - exported_files)
    extra = sorted(exported_files - expected_set)

    # detect collisions where multiple expected map to same filename
    filename_map = {}
    collisions = {}
    for model_id, name, filename, count in expected_files:
        filename_map.setdefault(filename, []).append((model_id, name))
        if len(filename_map[filename]) > 1:
            collisions[filename] = filename_map[filename]

    # Read exporter's ExportLog.txt mapping if available for extra verification
    export_log_path = os.path.join(EXPORT_ROOT, vox_base, "ExportLog.txt")
    exportlog_map = read_exportlog_mappings(export_log_path)

    report_path = os.path.join(REPORTS_DIR, f"CompareReport_{vox_base}.txt")
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write(f"CompareReport for {vox_file}\n\n")
        rep.write("Expected exported files (computed using ExportMeshes.py logic):\n")
        for model_id, name, filename, count in expected_files:
            rep.write(f"  model {model_id} -> {name} -> {filename}  ({count} voxels)\n")
        rep.write("\nActual exported files in folder:\n")
        if exported_files:
            for f in sorted(exported_files):
                rep.write(f"  {f}\n")
        else:
            rep.write("  (no exported files found)\n")

        rep.write("\nExportLog mappings (if present):\n")
        if exportlog_map:
            for fn, assigned in sorted(exportlog_map.items()):
                rep.write(f"  {fn} => {assigned}\n")
        else:
            rep.write("  (no ExportLog mappings found)\n")

        rep.write("\nSummary:\n")
        rep.write(f"  Expected count: {len(expected_set)}\n")
        rep.write(f"  Actual exported count: {len(exported_files)}\n")
        rep.write(f"  Missing files: {len(missing)}\n")
        for m in missing:
            rep.write(f"    - {m}\n")
        rep.write(f"  Extra files: {len(extra)}\n")
        for e in extra:
            rep.write(f"    - {e}\n")

        if collisions:
            rep.write("\nFilename collisions (multiple models map to same sanitized filename):\n")
            for fn, items in collisions.items():
                rep.write(f"  {fn} <- {items}\n")
            rep.write("\nWarning: collisions cause earlier exports to be overwritten by later ones.\n")

        # Cross-check expected name vs exporter's declared assigned name (if available)
        mismatched_names = []
        for _, expected_name, filename, _ in expected_files:
            declared = exportlog_map.get(filename)
            if declared and declared != expected_name:
                mismatched_names.append((filename, expected_name, declared))
        if mismatched_names:
            rep.write("\nName mismatches between expected mapping and ExportLog:\n")
            for fn, exp, decl in mismatched_names:
                rep.write(f"  {fn}: expected '{exp}'  but ExportLog declared '{decl}'\n")

        rep.write("\nNote: this tool mirrors ExportMeshes.py naming logic exactly. If you change ExportMeshes.py, keep this script updated.\n")

    # Console summary
    print(f"  Expected: {len(expected_set)}  Actual: {len(exported_files)}  Missing: {len(missing)}  Extra: {len(extra)}  Collisions: {len(collisions)}")
    if exportlog_map:
        print(f"  ExportLog mappings found: {len(exportlog_map)}  Name mismatches: {len(mismatched_names)}")
    print(f"  Report written to: {report_path}\n")