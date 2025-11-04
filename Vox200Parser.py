import os
import struct
import re
from collections import defaultdict, namedtuple

Voxel = namedtuple("Voxel", ["x", "y", "z", "color_index"])

class Vox200Parser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.voxels_by_layer = defaultdict(list)
        self.layer_name_map = {}           # Final mapping: raw_key -> descriptive name (for non-empty parts)
        self.raw_part_name_map = {}        # New: mapping for every Part_X (including empty/helper parts)
        self.layer_id_to_name = {}         # LAYR chunk: layer_id -> name
        self.layer_ids_in_order = []       # keep LAYR order as seen in file
        self.node_id_to_name = {}          # nTRN / nSHP: node_id or model_id -> name
        self.node_to_child = {}            # nTRN: transform_node_id -> child_node_id
        self.node_to_layer = {}            # nTRN: transform_node_id -> layer_id (if present)
        self.shape_node_to_models = {}     # nTRN/nSHP: shape_node_id -> [model_ids]
        self.model_id_to_shape = {}        # model_id -> shape_node_id

    def _parse_attr_dict_from(self, content, offset, count):
        """Read count attribute key/value pairs from content at offset. Returns (attrs_dict, new_offset)."""
        attrs = {}
        for _ in range(count):
            if offset + 4 > len(content): break
            key_len = struct.unpack("<I", content[offset:offset+4])[0]; offset += 4
            if offset + key_len > len(content): break
            key = content[offset:offset+key_len].decode("utf-8", errors="ignore"); offset += key_len
            if offset + 4 > len(content): break
            val_len = struct.unpack("<I", content[offset:offset+4])[0]; offset += 4
            if offset + val_len > len(content): break
            val = content[offset:offset+val_len].decode("utf-8", errors="ignore"); offset += val_len
            attrs[key] = val
        return attrs, offset

    def _sanitize_name(self, name):
        if not name:
            return ""
        # Replace problematic chars, collapse whitespace, trim
        n = re.sub(r"[^\w\s\.-]", "_", name).strip()
        n = re.sub(r"\s+", "_", n)
        return n

    def _looks_like_helper(self, name):
        if not name:
            return True
        ln = name.strip()
        if not ln:
            return True
        # helper heuristics: numeric-only, very short numeric, common helper tokens, or startswith underscore
        if ln.isdigit():
            return True
        if len(ln) <= 2 and any(c.isdigit() for c in ln):
            return True
        low = ln.lower()
        if low in ("00", "helper", "layer", "_name"):
            return True
        if ln.startswith("_"):
            return True
        return False

    def parse(self):
        with open(self.filepath, "rb") as f:
            if f.read(4) != b"VOX ":
                raise ValueError("Not a valid VOX file")
            version = struct.unpack("<I", f.read(4))[0]

            def read_chunk():
                chunk_id = f.read(4)
                if not chunk_id:
                    return None, None, None
                header = f.read(8)
                if len(header) < 8:
                    return None, None, None
                content_size, children_size = struct.unpack("<II", header)
                content = f.read(content_size)
                if len(content) < content_size:
                    # truncated but continue best-effort
                    pass
                return chunk_id, content, children_size

            part_counter = 0

            # First pass: collect LAYR in order (store raw content if needed), collect XYZI voxels and other node info
            while True:
                chunk_id, content, children_size = read_chunk()
                if chunk_id is None:
                    break

                if chunk_id == b"SIZE":
                    if len(content) < 12: continue

                elif chunk_id == b"XYZI":
                    if len(content) < 4: continue
                    num_voxels = struct.unpack("<I", content[:4])[0]
                    voxel_data = content[4:]
                    key = f"Part_{part_counter}"
                    for i in range(num_voxels):
                        seg = voxel_data[i*4:(i+1)*4]
                        if len(seg) < 4:
                            break
                        x, y, z, color_index = struct.unpack("BBBB", seg)
                        self.voxels_by_layer[key].append(Voxel(x, y, z, color_index))
                    part_counter += 1

                elif chunk_id == b"nTRN":
                    if len(content) < 8: continue
                    node_id = struct.unpack("<I", content[:4])[0]
                    attr_dict_len = struct.unpack("<I", content[4:8])[0]
                    offset = 8
                    attrs, offset = self._parse_attr_dict_from(content, offset, attr_dict_len)
                    name = attrs.get("_name")
                    # try to read child node id if present (best-effort)
                    child_node_id = None
                    if offset + 4 <= len(content):
                        try:
                            child_node_id = struct.unpack("<i", content[offset:offset+4])[0]
                            self.node_to_child[node_id] = child_node_id
                        except Exception:
                            child_node_id = None
                    # Try to read layer id if present at common offset (best-effort)
                    # Some exporters put layer_id further in the content (heuristic)
                    try:
                        if len(content) >= 28:
                            layer_candidate = struct.unpack("<I", content[24:28])[0]
                            # store only if plausible (non-negative)
                            self.node_to_layer[node_id] = layer_candidate
                    except Exception:
                        pass
                    if name:
                        self.node_id_to_name[node_id] = name

                elif chunk_id == b"nGRP":
                    continue

                elif chunk_id == b"nSHP":
                    if len(content) < 8: continue
                    offset = 0
                    shape_node_id = struct.unpack("<I", content[offset:offset+4])[0]; offset += 4
                    model_count = struct.unpack("<I", content[offset:offset+4])[0]; offset += 4
                    models = []
                    for _ in range(model_count):
                        if offset + 4 > len(content): break
                        model_id = struct.unpack("<I", content[offset:offset+4])[0]; offset += 4
                        models.append(model_id)
                    if models:
                        self.shape_node_to_models.setdefault(shape_node_id, []).extend(models)
                        for m in models:
                            self.model_id_to_shape[m] = shape_node_id
                    # try reading small attr dict if present
                    if offset + 4 <= len(content):
                        try:
                            num_attrs = struct.unpack("<I", content[offset:offset+4])[0]; offset += 4
                            attrs, _ = self._parse_attr_dict_from(content, offset, num_attrs)
                            name = attrs.get("_name")
                            if name:
                                self.node_id_to_name[shape_node_id] = name
                        except Exception:
                            pass

                elif chunk_id == b"LAYR":
                    if len(content) < 4: continue
                    layer_id = struct.unpack("<I", content[:4])[0]
                    # Try standard layout first (attr_dict_len at offset 8)
                    name = None
                    if len(content) >= 12:
                        try:
                            attr_dict_len = struct.unpack("<I", content[8:12])[0]
                            offset = 12
                            attrs, _ = self._parse_attr_dict_from(content, offset, attr_dict_len)
                            name = attrs.get("_name")
                        except Exception:
                            name = None
                    # If not found, try offset-4 heuristic (some exporters)
                    if not name:
                        if len(content) >= 8:
                            try:
                                offset = 4
                                attr_dict_len = struct.unpack("<I", content[offset:offset+4])[0]
                                offset += 4
                                attrs, _ = self._parse_attr_dict_from(content, offset, attr_dict_len)
                                name = attrs.get("_name")
                            except Exception:
                                name = None
                    if name:
                        self.layer_id_to_name[layer_id] = name
                        self.layer_ids_in_order.append(layer_id)

                elif chunk_id == b"RGBA":
                    continue

            # --- Build ordered_layer_names (skip "00") same way ExportMeshes.py did previously ---
            ordered_layer_names = []
            ordered_layer_ids = []
            for lid in self.layer_ids_in_order:
                lname = self.layer_id_to_name.get(lid, f"Layer_{lid}")
                if lname != "00":
                    ordered_layer_names.append(lname)
                    ordered_layer_ids.append(lid)

            # Prepare helper maps to reason about whether a node name maps to voxel-containing models
            # Build a set of model indices that have voxels (Part_N where N matches model index)
            model_indices_with_vox = set()
            for raw_k, voxels in self.voxels_by_layer.items():
                try:
                    idx = int(raw_k.split("_", 1)[1])
                    if len(voxels) > 0:
                        model_indices_with_vox.add(idx)
                except Exception:
                    continue

            # Build a map shape_node -> has_vox (if any of its models are in model_indices_with_vox)
            shape_has_vox = {}
            for shape_node, models in self.shape_node_to_models.items():
                has = any(m in model_indices_with_vox for m in models)
                shape_has_vox[shape_node] = has

            # --- Final robust mapping: prefer model->shape->transform->_name, but avoid helper nodes and ensure uniqueness ---
            final_map = {}
            used_names = set()
            # Build list of non-empty Part keys in XYZI order
            nonempty_parts = [k for k,v in self.voxels_by_layer.items() if len(v) > 0]

            for idx, raw_key in enumerate(nonempty_parts):
                # determine model index number from the raw_key "Part_N"
                try:
                    model_idx = int(raw_key.split("_",1)[1])
                except Exception:
                    model_idx = idx

                chosen = None
                reason = None

                # 1) model -> shape node
                shape_node = self.model_id_to_shape.get(model_idx)
                if shape_node is not None:
                    # find transform nodes that point to this shape (node_to_child)
                    transform_nodes = [tn for tn, child in self.node_to_child.items() if child == shape_node]
                    if transform_nodes:
                        # prefer a transform node name
                        for tn in transform_nodes:
                            tn_name = self.node_id_to_name.get(tn)
                            if tn_name:
                                chosen = tn_name
                                reason = f"nTRN name from transform node {tn} (points to shape {shape_node})"
                                break
                        # if no tn name, try transform->layer->LAYR
                        if not chosen:
                            for tn in transform_nodes:
                                layer_id = self.node_to_layer.get(tn)
                                if layer_id is not None:
                                    layr_name = self.layer_id_to_name.get(layer_id)
                                    if layr_name:
                                        chosen = layr_name
                                        reason = f"LAYR name via transform node {tn} -> layer {layer_id}"
                                        break

                # 2) direct shape node name
                if not chosen:
                    shape_named = self.node_id_to_name.get(shape_node)
                    if shape_named:
                        chosen = shape_named
                        reason = f"nSHP / shape node name for shape {shape_node}"

                # 3) fallback to LAYR mapping by ordered non-empty index (previous behavior)
                if not chosen:
                    # use the idx'th ordered_layer_names if available
                    if idx < len(ordered_layer_names):
                        chosen = ordered_layer_names[idx]
                        reason = f"LAYR order fallback index {idx}"
                    else:
                        chosen = raw_key
                        reason = "fallback raw key"

                final_map[raw_key] = chosen

            self.layer_name_map = final_map

            # --- NEW: populate raw_part_name_map for every Part_X (including empty/helper parts) ---
            # Determine full set of model indices we can reason about:
            all_model_indices = set(model_indices_with_vox)
            all_model_indices.update(self.model_id_to_shape.keys())
            # Also include any Part_ keys present in voxels_by_layer even if empty (defensive)
            for raw_k in list(self.voxels_by_layer.keys()):
                try:
                    i = int(raw_k.split("_",1)[1])
                    all_model_indices.add(i)
                except Exception:
                    pass

            raw_map = {}
            for model_idx in sorted(all_model_indices):
                raw_key = f"Part_{model_idx}"
                chosen = None

                # 1) model -> shape node
                shape_node = self.model_id_to_shape.get(model_idx)
                if shape_node is not None:
                    transform_nodes = [tn for tn, child in self.node_to_child.items() if child == shape_node]
                    if transform_nodes:
                        for tn in transform_nodes:
                            tn_name = self.node_id_to_name.get(tn)
                            if tn_name:
                                chosen = tn_name
                                break
                        if not chosen:
                            for tn in transform_nodes:
                                layer_id = self.node_to_layer.get(tn)
                                if layer_id is not None:
                                    layr_name = self.layer_id_to_name.get(layer_id)
                                    if layr_name:
                                        chosen = layr_name
                                        break

                # 2) direct shape node name
                if not chosen:
                    shape_named = self.node_id_to_name.get(shape_node)
                    if shape_named:
                        chosen = shape_named

                # 3) fallback to LAYR ordered names by model index if possible
                if not chosen:
                    if model_idx < len(ordered_layer_names):
                        chosen = ordered_layer_names[model_idx]
                    else:
                        # last resort: if this part had voxels, prefer the previously computed final_map entry
                        if raw_key in final_map:
                            chosen = final_map[raw_key]
                        else:
                            chosen = raw_key

                raw_map[raw_key] = chosen

            self.raw_part_name_map = raw_map
            # --- END NEW population ---

            # Write detailed DebugNameMap_<voxname>.txt in reports/
            try:
                reports_dir = os.path.join(os.path.dirname(__file__), "reports")
                os.makedirs(reports_dir, exist_ok=True)
                vox_base = os.path.splitext(os.path.basename(self.filepath))[0]
                debug_path = os.path.join(reports_dir, f"DebugNameMap_{vox_base}.txt")
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(f"DebugNameMap for: {os.path.basename(self.filepath)}\n\n")

                    f.write("Layer ID -> Name (LAYR):\n")
                    if self.layer_id_to_name:
                        for lid, lname in sorted(self.layer_id_to_name.items()):
                            f.write(f"  [{lid}] => {lname}\n")
                    else:
                        f.write("  (no LAYR entries found)\n")
                    f.write("\n")

                    f.write("Node ID -> Name (nTRN / nSHP heuristics):\n")
                    if self.node_id_to_name:
                        for nid, nname in sorted(self.node_id_to_name.items()):
                            f.write(f"  [{nid}] => {nname}\n")
                    else:
                        f.write("  (no named nodes found)\n")
                    f.write("\n")

                    f.write("Shape Node -> Model IDs (nSHP):\n")
                    if self.shape_node_to_models:
                        for sid, models in sorted(self.shape_node_to_models.items()):
                            f.write(f"  ShapeNode {sid} => Models: {models}\n")
                    else:
                        f.write("  (no shape->model mappings found)\n")
                    f.write("\n")

                    f.write("Transform Node -> Child Node (nTRN):\n")
                    if self.node_to_child:
                        for tn, child in sorted(self.node_to_child.items()):
                            f.write(f"  TransformNode {tn} => ChildNode {child}\n")
                    else:
                        f.write("  (no transform->child mappings found)\n")
                    f.write("\n")

                    f.write("Transform Node -> Layer ID (nTRN heuristics):\n")
                    if self.node_to_layer:
                        for tn, lid in sorted(self.node_to_layer.items()):
                            f.write(f"  TransformNode {tn} => LayerID {lid}\n")
                    else:
                        f.write("  (no transform->layer mappings found)\n")
                    f.write("\n")

                    f.write("Final assignment (Part_X -> chosen name) for non-empty parts (layer_name_map):\n")
                    for raw_key, chosen in self.layer_name_map.items():
                        count = len(self.voxels_by_layer.get(raw_key, []))
                        f.write(f"  {raw_key} ({count} voxels) => {chosen}\n")
                    f.write("\n")

                    f.write("Raw part name map (all Part_X keys, including empty/helper parts):\n")
                    for raw_key, chosen in sorted(self.raw_part_name_map.items()):
                        count = len(self.voxels_by_layer.get(raw_key, []))
                        f.write(f"  {raw_key} ({count} voxels) => {chosen}\n")

                    f.write("\nEnd of DebugNameMap\n")
            except Exception:
                pass

            return self
