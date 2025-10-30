import os
import sys
import numpy as np
import trimesh
import struct
from collections import defaultdict, namedtuple
from Vox200Parser import Vox200Parser

Voxel = namedtuple("Voxel", ["x", "y", "z", "color_index"])

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in (name or ""))

BASE_DIR = os.path.dirname(__file__)
EXPORT_DIR = os.path.join(BASE_DIR, "exported_meshes")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]
if not vox_files:
    print("No .vox files found.")
    sys.exit(0)

for vox_file in vox_files:
    vox_path = os.path.join(BASE_DIR, vox_file)
    vox_name = os.path.splitext(vox_file)[0]
    sub_export = os.path.join(EXPORT_DIR, vox_name)
    os.makedirs(sub_export, exist_ok=True)

    parser = Vox200Parser(vox_path).parse()
    layer_voxels = parser.voxels_by_layer
    layer_name_map = parser.layer_name_map

    export_log_path = os.path.join(REPORTS_DIR, f"{vox_name}_ExportLog.txt")
    with open(export_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Assigned Layer Names:\n")
        for raw_key, name in layer_name_map.items():
            log_file.write(f"{raw_key} => {name}\n")

        log_file.write("\nExport Status:\n")
        for idx, (raw_key, voxels) in enumerate(layer_voxels.items()):
            layer_name = layer_name_map.get(raw_key, f"Unnamed_{idx}")
            safe_name = sanitize_filename(layer_name) or f"Part_{idx}"

            if not voxels:
                log_file.write(f"Skipping empty layer: {safe_name}\n")
                continue

            positions = np.array([[v.x, v.y, v.z] for v in voxels])
            cubes = []
            for pos in positions:
                cube = trimesh.creation.box(extents=(1, 1, 1))
                cube.apply_translation(pos)
                cubes.append(cube)

            combined = trimesh.util.concatenate(cubes)
            export_path = os.path.join(sub_export, f"{safe_name}.obj")
            combined.export(export_path)

            log_file.write(f"Exported: {export_path} with {len(voxels)} voxels\n")

    print(f"Export complete for {vox_file}. See {export_log_path}.")
