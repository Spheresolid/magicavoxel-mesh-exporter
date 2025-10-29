# magicavoxel-mesh-exporter
MagicaVoxel Layer-Based OBJ Export Tool

This Python utility is built for extracting and exporting layered 3D mesh data from .vox files created in MagicaVoxel (particularly VOX 200 format). It identifies and separates voxel model components by named layers or transformation nodes, and exports each part as a properly named .obj mesh file instead of generic names like Part_1.obj.

🎯 Purpose

MagicaVoxel supports layer naming and a scene graph hierarchy. However, when exporting .vox files, many tools fail to preserve those names. Instead, they dump the entire scene into a single object or export unlabeled parts.
This tool solves that problem by:
Parsing .vox files to extract explicit part names (like head, left_arm, body, etc.)
Mapping those names to corresponding voxel mesh data
Exporting each part as an individual .obj file using its proper name
Generating detailed logs of the export and internal hierarchy for debugging
This makes the tool ideal for character animation pipelines, game dev asset prep, rigging workflows, or any case where model parts need clean, semantic separation.

💡 Features

✅ Parses LAYR, nTRN, and nSHP chunks from .vox to map layers to names
✅ Exports each model part as a separate .obj mesh
✅ Assigns correct filenames (e.g., head.obj, Right_Arm.obj) based on internal metadata
✅ Includes fallback names for unnamed parts
✅ Outputs detailed ExportLog.txt and DebugNameMap.txt for troubleshooting
✅ Also includes CheckEmptyLayers.py to scan for empty layers

project_root/
│
├── sourcefile.vox               # Source MagicaVoxel file
├── ExportMeshes.py              # Main script to export .obj files
├── CheckEmptyLayers.py          # Utility to detect empty layers
├── exported_meshes/             # Auto-created output folder
│   ├── head.obj
│   ├── Right_Arm.obj
│   └── ...
└── reports/
    ├── ExportLog.txt            # Log of exports, names, voxel counts
    └── DebugNameMap.txt         # Deep mapping report of all nodes/layers
    
🛠 Requirements
Python 3.8+
trimesh
numpy

Install with:

bash
pip install numpy trimesh

How to Use:
1. Place your .vox file (VOX 200 format) in the same directory (or change the filename inside the script).
2. Run the export script:

bash
python ExportMeshes.py

3. Check exported_meshes/ for the resulting .obj files.
4. Review ExportLog.txt for summaries and reports/DebugNameMap.txt for full hierarchy info.

You can also run the layer checker separately:

bash
python CheckEmptyLayers.py

⚙️ How It Works

The parser reads binary chunks from the VOX file: LAYR, nTRN, nSHP, and XYZI
It constructs a scene graph and matches model IDs to names via transform and layer ID mapping
It uses Trimesh to build meshes per voxel cluster
Final names are assigned via this priority:

1. nTRN transform node _name
2. LAYR name
3. Fallback to Unnamed_<index>

🔍 Debugging and Customization

The script creates:

ExportLog.txt: Lists all exports and names
DebugNameMap.txt: Shows model ID mappings, transform node links, and raw names
These are essential if you're trying to ensure the correct part is named (e.g., part_3 is really the head).

📦 Future-Proof Design:
Your models may evolve over time with more layers or name changes. This tool is built to preserve all detected names, even if some are not used in the final export. This keeps the data useful for future .vox files with similar structure.
