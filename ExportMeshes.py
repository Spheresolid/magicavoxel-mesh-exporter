import os
import struct
import numpy as np
import trimesh
from collections import namedtuple

Voxel = namedtuple("Voxel", ["x", "y", "z", "color_index"])

def sanitize_filename(name):
    return ''.join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in name).strip()

def parse_vox_layers_and_models(filepath):
    with open(filepath, "rb") as f:
        if f.read(4) != b"VOX ":
            raise ValueError("Not a valid VOX file")
        version = struct.unpack("<I", f.read(4))[0]

        layer_id_to_name = {}
        model_voxels = {}
        model_voxel_counts = {}
        layer_ids_in_order = []

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
                attr_dict_len = struct.unpack("<I", content[offset:offset+4])[0]
                offset += 4
                name = None
                for _ in range(attr_dict_len):
                    if offset + 4 > len(content): break
                    key_len = struct.unpack("<I", content[offset:offset+4])[0]
                    offset += 4
                    key = content[offset:offset+key_len].decode("utf-8", errors="ignore")
                    offset += key_len
                    if offset + 4 > len(content): break
                    val_len = struct.unpack("<I", content[offset:offset+4])[0]
                    offset += 4
                    if offset + val_len > len(content): break
                    val = content[offset:offset+val_len].decode("utf-8", errors="ignore")
                    offset += val_len
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

        # Build ordered list of non-"00" layer names
        ordered_layer_names = []
        ordered_layer_ids = []
        for lid in layer_ids_in_order:
            lname = layer_id_to_name.get(lid, f"Layer_{lid}")
            if lname != "00":
                ordered_layer_names.append(lname)
                ordered_layer_ids.append(lid)

        return model_voxels, ordered_layer_names, model_voxel_counts

BASE_DIR = os.path.dirname(__file__)
EXPORT_DIR = os.path.join(BASE_DIR, "exported_meshes")

if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# Find all .vox files in the folder
vox_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".vox")]

if not vox_files:
    print("No .vox files found.")
else:
    for vox_file in vox_files:
        vox_path = os.path.join(BASE_DIR, vox_file)
        vox_name = os.path.splitext(vox_file)[0]
        sub_export_dir = os.path.join(EXPORT_DIR, vox_name)
        if not os.path.exists(sub_export_dir):
            os.makedirs(sub_export_dir)

        model_voxels, ordered_layer_names, model_voxel_counts = parse_vox_layers_and_models(vox_path)
        export_log_path = os.path.join(sub_export_dir, "ExportLog.txt")

        with open(export_log_path, "w") as log_file:
            log_file.write(f"Export Status & Layer Name Mapping (by index, skipping '00') for {vox_file}:\n\n")
            idx = 0
            for model_id in sorted(model_voxels.keys()):
                voxels = model_voxels[model_id]
                voxel_count = len(voxels)
                if voxel_count == 0:
                    log_file.write(f"Skipping empty model: {model_id}\n")
                    continue

                # Assign name by order, skipping '00'
                if idx < len(ordered_layer_names):
                    name = ordered_layer_names[idx]
                else:
                    name = f"Part_{model_id}"
                safe_name = sanitize_filename(name)
                idx += 1

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
                    log_file.write(f"Exported: {export_path} as {name} with {voxel_count} voxels\n")
                else:
                    log_file.write(f"Model {model_id} ({safe_name}) had no cubes to export\n")

            log_file.write("\nOrdered Layer Names Used:\n")
            for i, n in enumerate(ordered_layer_names):
                log_file.write(f"  [{i}] {n}\n")
            log_file.write("\nModel voxel counts:\n")
            for model_id, count in model_voxel_counts.items():
                log_file.write(f"  model {model_id}: {count} voxels\n")

        print(f"Export complete for {vox_file}. See {export_log_path} for details.")