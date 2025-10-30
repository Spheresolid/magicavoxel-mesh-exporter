import os
import struct
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
        _ = struct.unpack("<I", f.read(4))[0]

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
                    x, y, z, ci = struct.unpack("BBBB", voxel_data[i * 4:(i + 1) * 4])
                    voxels.append((x,y,z,ci))
                model_voxels[model_counter] = voxels
                model_voxel_counts[model_counter] = len(voxels)
                model_counter += 1

    ordered_layer_names = []
    for lid in layer_ids_in_order:
        lname = layer_id_to_name.get(lid, f"Layer_{lid}")
        if lname != "00":
            ordered_layer_names.append(lname)

    return model_voxels, ordered_layer_names, model_voxel_counts

def print_assignments(vox_path):
    model_voxels, ordered_layer_names, model_voxel_counts = parse_vox_layers_and_models(vox_path)
    print(f"Parsed: {os.path.basename(vox_path)}")
    print(f"Ordered LAYR names ({len(ordered_layer_names)}): {ordered_layer_names}\n")

    idx = 0
    for model_id in sorted(model_voxels.keys()):
        voxels = model_voxels[model_id]
        if len(voxels) == 0:
            print(f"model {model_id}: EMPTY -> skipped")
            continue
        if idx < len(ordered_layer_names):
            name = ordered_layer_names[idx]
        else:
            name = f"Part_{model_id}"
        safe = sanitize_filename(name)
        print(f"model {model_id} -> raw Part_{model_id} -> assigned '{name}' -> file '{safe}.obj' -> {len(voxels)} voxels")
        idx += 1

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    vox_files = [f for f in os.listdir(base) if f.lower().endswith('.vox')]
    if not vox_files:
        print("No .vox files found in folder.")
    else:
        # choose Character.vox explicitly if present
        target = 'Character.vox' if 'Character.vox' in vox_files else vox_files[0]
        print_assignments(os.path.join(base, target))