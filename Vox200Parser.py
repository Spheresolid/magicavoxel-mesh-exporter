import struct
from collections import defaultdict, namedtuple

Voxel = namedtuple("Voxel", ["x", "y", "z", "color_index"])

class Vox200Parser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.voxels_by_layer = defaultdict(list)
        self.layer_name_map = {} # This will store the final mapping
        self.layer_id_to_name = {} # Temporary store for LAYR chunk names
        self.node_id_to_name = {} # Temporary store for nTRN/nSHP chunk names

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
                if len(header) < 8: # Ensure enough bytes for header
                    return None, None, None
                content_size, children_size = struct.unpack("<II", header)
                content = f.read(content_size)
                # Ensure content is fully read, or handle if file ends prematurely
                if len(content) < content_size:
                    print(f"Warning: Chunk {chunk_id.decode()} content truncated. Expected {content_size}, got {len(content)}")
                return chunk_id, content, children_size

            part_counter = 0 # To assign generic Part_X keys to XYZI chunks

            while True:
                chunk_id, content, children_size = read_chunk()
                if chunk_id is None:
                    break

                if chunk_id == b"SIZE":
                    if len(content) < 12: continue
                    size_x, size_y, size_z = struct.unpack("<III", content)
                elif chunk_id == b"XYZI":
                    if len(content) < 4: continue
                    num_voxels = struct.unpack("<I", content[:4])[0]
                    voxel_data = content[4:]
                    
                    key = f"Part_{part_counter}" 
                    for i in range(num_voxels):
                        if len(voxel_data[i * 4:(i + 1) * 4]) < 4: # Check for complete voxel data
                            break # Incomplete voxel, stop processing this chunk
                        x, y, z, color_index = struct.unpack("BBBB", voxel_data[i * 4:(i + 1) * 4])
                        self.voxels_by_layer[key].append(Voxel(x, y, z, color_index))
                    part_counter += 1 # Increment for the next XYZI chunk

                elif chunk_id == b"RGBA":
                    # Color palette data, not directly used for naming layers
                    pass
                
                elif chunk_id == b"nTRN": # Transform Node
                    if len(content) < 8: continue # node_id and attr_dict_len
                    node_id = struct.unpack("<I", content[:4])[0]
                    attr_dict_len = struct.unpack("<I", content[4:8])[0]
                    offset = 8
                    name = None
                    for _ in range(attr_dict_len):
                        if offset + 4 > len(content): break # Safety check for key_len
                        key_len = struct.unpack("<I", content[offset:offset+4])[0]
                        offset += 4
                        
                        if offset + key_len > len(content): break # Safety check for key
                        try:
                            key = content[offset:offset+key_len].decode("utf-8", errors="ignore")
                        except UnicodeDecodeError:
                            key = ""
                        offset += key_len
                        
                        if offset + 4 > len(content): break # Safety check for val_len
                        val_len = struct.unpack("<I", content[offset:offset+4])[0]
                        offset += 4
                        
                        if offset + val_len > len(content): break # Safety check for val
                        try:
                            val = content[offset:offset+val_len].decode("utf-8", errors="ignore")
                        except UnicodeDecodeError:
                            val = ""
                        offset += val_len
                        if key == "_name":
                            name = val
                    
                    if offset + 4 <= len(content): # Check before accessing child_node_id
                        child_node_id = struct.unpack("<i", content[offset:offset+4])[0]
                        if name: # Only store if a name was found
                            self.node_id_to_name[node_id] = name 

                elif chunk_id == b"nGRP": # Group Node
                    if len(content) < 12: continue # node_id, num_children
                    node_id = struct.unpack("<I", content[:4])[0]
                    num_children = struct.unpack("<I", content[8:12])[0]
                    offset = 12
                    children_ids = []
                    for _ in range(num_children):
                        if offset + 4 > len(content): break # Safety check for child ID
                        children_ids.append(struct.unpack("<I", content[offset:offset+4])[0])
                        offset += 4

                elif chunk_id == b"nSHP": # Shape Node
                    if len(content) < 12: continue # node_id, model_id, num_attrs
                    node_id = struct.unpack("<I", content[:4])[0]
                    model_id = struct.unpack("<I", content[4:8])[0] # Model ID (index into XYZI chunks)
                    num_attrs = struct.unpack("<I", content[8:12])[0]
                    offset = 12
                    name = None
                    for _ in range(num_attrs):
                        if offset + 4 > len(content): break # Safety check for key_len
                        key_len = struct.unpack("<I", content[offset:offset+4])[0]
                        offset += 4
                        
                        if offset + key_len > len(content): break # Safety check for key
                        try:
                            key = content[offset:offset+key_len].decode("utf-8", errors="ignore")
                        except UnicodeDecodeError:
                            key = ""
                        offset += key_len
                        
                        if offset + 4 > len(content): break # Safety check for val_len
                        val_len = struct.unpack("<I", content[offset:offset+4])[0]
                        offset += 4
                        
                        if offset + val_len > len(content): break # Safety check for val
                        try:
                            val = content[offset:offset+val_len].decode("utf-8", errors="ignore")
                        except UnicodeDecodeError:
                            val = ""
                        offset += val_len
                        if key == "_name":
                            name = val
                    if name:
                        self.node_id_to_name[model_id] = name # Map model_id to its name

                elif chunk_id == b"LAYR": # Layer Node
                    if len(content) < 12: continue # layer_id, reserved, attr_dict_len
                    layer_id = struct.unpack("<I", content[:4])[0]
                    # skip 4 reserved bytes: content[4:8]
                    attr_dict_len = struct.unpack("<I", content[8:12])[0]
                    offset = 12
                    name = None
                    for _ in range(attr_dict_len):
                        if offset + 4 > len(content): break # Safety check for key_len
                        key_len = struct.unpack("<I", content[offset:offset+4])[0]
                        offset += 4
                        
                        if offset + key_len > len(content): break # Safety check for key
                        try:
                            key = content[offset:offset+key_len].decode("utf-8", errors="ignore")
                        except UnicodeDecodeError:
                            key = ""
                        offset += key_len
                        
                        if offset + 4 > len(content): break # Safety check for val_len
                        val_len = struct.unpack("<I", content[offset:offset+4])[0]
                        offset += 4
                        
                        if offset + val_len > len(content): break # Safety check for val
                        try:
                            val = content[offset:offset+val_len].decode("utf-8", errors="ignore")
                        except UnicodeDecodeError:
                            val = ""
                        offset += val_len
                        if key == "_name":
                            name = val
                    if name:
                        self.layer_id_to_name[layer_id] = name

            # --- Final mapping logic ---
            final_mapping = {}
            for i, raw_key in enumerate(self.voxels_by_layer.keys()): # 'Part_0', 'Part_1', etc.
                descriptive_name = self.node_id_to_name.get(i) or self.layer_id_to_name.get(i)
                if descriptive_name:
                    final_mapping[raw_key] = descriptive_name
                else:
                    final_mapping[raw_key] = raw_key # Fallback to Part_X if no name found
            
            self.layer_name_map = final_mapping
            return self