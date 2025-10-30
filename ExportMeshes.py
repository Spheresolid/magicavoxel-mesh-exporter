# ExportMeshes.py
import os
import numpy as np
import trimesh
from Vox200Parser import Vox200Parser

def sanitize_filename(name):
    return ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in (name or "")).strip()

BASE_DIR = os.path.dirname(__file__)
os.chdir(BASE_DIR)

EXPORT_ROOT = os.path.join(BASE_DIR, "exported_meshes")
os.makedirs(EXPORT_ROOT, exist_ok=True)

# Find all .vox files in the folder
vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]

if not vox_files:
    print("No .vox files found.")
else:
    for vox_file in vox_files:
        vox_path = os.path.join(BASE_DIR, vox_file)
        vox_name = os.path.splitext(vox_file)[0]
        sub_export_dir = os.path.join(EXPORT_ROOT, vox_name)
        os.makedirs(sub_export_dir, exist_ok=True)

        # Use the shared parser to obtain voxels and final name mapping (Part_X -> assigned name)
        parser = Vox200Parser(vox_path).parse()
        voxels_by_layer = parser.voxels_by_layer       # Ordered by XYZI chunk order
        name_map = parser.layer_name_map               # raw_key -> chosen descriptive name

        export_log_path = os.path.join(sub_export_dir, "ExportLog.txt")
        with open(export_log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Export Status & Layer Name Mapping for {vox_file}:\n\n")

            for raw_key, voxels in voxels_by_layer.items():
                voxel_count = len(voxels)
                if voxel_count == 0:
                    log_file.write(f"Skipping empty model: {raw_key}\n")
                    continue

                # Use parser-provided mapping (1:1 Part_X -> name). Fallback to raw_key if not present.
                assigned_name = name_map.get(raw_key) or raw_key
                safe_name = sanitize_filename(assigned_name) or raw_key

                positions = np.array([[v.x, v.y, v.z] for v in voxels])
                cubes = []
                for pos in positions:
                    cube = trimesh.creation.box(extents=(1, 1, 1))
                    cube.apply_translation(pos)
                    cubes.append(cube)

                if cubes:
                    combined = trimesh.util.concatenate(cubes)
                    export_path = os.path.join(sub_export_dir, f"{safe_name}.obj")
                    combined.export(export_path)
                    log_file.write(f"Exported: {export_path} as {assigned_name} with {voxel_count} voxels\n")
                else:
                    log_file.write(f"Model {raw_key} ({safe_name}) had no cubes to export\n")

            # Write summary of mapping used
            log_file.write("\nFinal Part -> Assigned Name mapping:\n")
            for rk, assigned in name_map.items():
                count = len(voxels_by_layer.get(rk, []))
                log_file.write(f"  {rk} ({count} voxels) => {assigned}\n")

            log_file.write("\nModel voxel counts (by Part_X):\n")
            for rk, voxels in voxels_by_layer.items():
                log_file.write(f"  {rk}: {len(voxels)} voxels\n")

        print(f"Export complete for {vox_file}. See {export_log_path} for details.")