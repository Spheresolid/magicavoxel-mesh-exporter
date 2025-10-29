import os
import sys
import numpy as np
import trimesh
import struct
from collections import defaultdict, namedtuple

Voxel = namedtuple("Voxel", ["x", "y", "z", "color_index"])

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

class Vox200Parser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.voxels_by_layer = defaultdict(list)
        self.layer_name_map = {}

    def parse(self):
        with open(self.filepath, "rb") as f:
            if f.read(4) != b"VOX ":
                raise ValueError("Not a valid VOX file")
            version = struct.unpack("<I", f.read(4))[0]

            node_to_name = {}
            model_id_to_node = {}
            node_links = {}
            shape_node_ids = set()
            model_counter = 0
            layer_id_to_name = {}
            node_to_layer = {}
            model_names_by_index = {}
            detected_names = []

            def read_chunk():
                chunk_id = f.read(4)
                if not chunk_id:
                    return None, None, None
                header = f.read(8)
                if len(header) < 8:
                    return None, None, None
                content_size, children_size = struct.unpack("<II", header)
                content = f.read(content_size)
                return chunk_id, content, children_size

            while True:
                chunk_id, content, _ = read_chunk()
                if chunk_id is None:
                    break

                if chunk_id == b"LAYR":
                    if len(content) < 12:
                        continue
                    layer_id = struct.unpack("<I", content[:4])[0]
                    attr_dict_len = struct.unpack("<I", content[8:12])[0]
                    offset = 12
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
                            detected_names.append(name)
                    if name:
                        layer_id_to_name[layer_id] = name

                elif chunk_id == b"nTRN":
                    node_id = struct.unpack("<I", content[:4])[0]
                    child_id = struct.unpack("<i", content[20:24])[0]
                    node_links[child_id] = node_id
                    offset = 12
                    name = None
                    if offset + 4 <= len(content):
                        dict_len = struct.unpack("<I", content[offset:offset+4])[0]
                        offset += 4
                        for _ in range(dict_len):
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
                                detected_names.append(name)
                    node_to_name[node_id] = name
                    if len(content) >= 28:
                        layer_id = struct.unpack("<I", content[24:28])[0]
                        node_to_layer[node_id] = layer_id

                elif chunk_id == b"nSHP":
                    shape_node_id = struct.unpack("<I", content[:4])[0]
                    model_count = struct.unpack("<I", content[4:8])[0]
                    offset = 8
                    for _ in range(model_count):
                        if offset + 4 > len(content): break
                        model_id = struct.unpack("<I", content[offset:offset+4])[0]
                        model_id_to_node[model_id] = shape_node_id
                        shape_node_ids.add(shape_node_id)
                        offset += 8

                elif chunk_id == b"XYZI":
                    if len(content) < 4:
                        continue
                    num_voxels = struct.unpack("<I", content[:4])[0]
                    voxel_data = content[4:]
                    key = f"Part_{model_counter}"
                    for i in range(num_voxels):
                        if i * 4 + 4 > len(voxel_data):
                            break
                        x, y, z, color_index = struct.unpack("BBBB", voxel_data[i * 4:(i + 1) * 4])
                        self.voxels_by_layer[key].append(Voxel(x, y, z, color_index))
                    model_counter += 1

        final_map = {}
        fallback_names = iter(detected_names)
        for model_id, part_key in zip(range(model_counter), self.voxels_by_layer.keys()):
            shape_node_id = model_id_to_node.get(model_id)
            name = None
            if shape_node_id is not None:
                transform_node = node_links.get(shape_node_id)
                if transform_node is not None:
                    name = node_to_name.get(transform_node)
                if not name:
                    layer_id = node_to_layer.get(transform_node)
                    name = layer_id_to_name.get(layer_id)
            if not name:
                try:
                    name = next(fallback_names)
                except StopIteration:
                    name = part_key
            final_map[part_key] = name

        self.layer_name_map = final_map
        self._debug_map(node_to_name, model_id_to_node, final_map)
        return self

    def _debug_map(self, node_to_name, model_id_to_node, final_map):
        debug_path = os.path.join(os.path.dirname(__file__), "reports", "DebugNameMap.txt")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w") as f:
            f.write("Node to Name Mapping:\n")
            for k, v in node_to_name.items():
                f.write(f"  Node {k}: {v}\n")

            f.write("\nModel ID to Shape Node Mapping:\n")
            for k, v in model_id_to_node.items():
                f.write(f"  Model {k} => Shape Node {v}\n")

            f.write("\nFinal Name Map:\n")
            for k, v in final_map.items():
                f.write(f"  {k} => {v}\n")

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exported_meshes")
VOX_FILE = os.path.join(os.path.dirname(__file__), "Trump.vox")

if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

parser = Vox200Parser(VOX_FILE).parse()
layer_voxels = parser.voxels_by_layer
layer_name_map = parser.layer_name_map

export_log_path = os.path.join(EXPORT_DIR, "ExportLog.txt")

with open(export_log_path, "w") as log_file:
    log_file.write("Assigned Layer Names:\n")
    for raw_key, name in layer_name_map.items():
        log_file.write(f"{raw_key} => {name}\n")

    log_file.write("\nExport Status:\n")
    for idx, (raw_key, voxels) in enumerate(layer_voxels.items()):
        layer_name = layer_name_map.get(raw_key, f"Unnamed_{idx}")
        safe_name = sanitize_filename(layer_name)

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
        export_path = os.path.join(EXPORT_DIR, f"{safe_name}.obj")
        combined.export(export_path)

        log_file.write(f"Exported: {export_path} with {len(voxels)} voxels\n")

check_script_path = os.path.join(os.path.dirname(__file__), "CheckEmptyLayers.py")
if os.path.exists(check_script_path):
    os.system(f"python \"{check_script_path}\" \"{VOX_FILE}\" \"{EXPORT_DIR}\"")
else:
    print("CheckEmptyLayers.py not detected; skipping it.")
