import os
import glob
from Vox200Parser import Vox200Parser

# Ensure we're in the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create and store reports in a separate folder
REPORT_DIR = os.path.abspath("reports")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORT_DIR, "EmptyLayerReport.txt")

def main():
    vox_files = glob.glob("*.vox")
    if not vox_files:
        print("No .vox files found in the script directory.")
        return

    input_vox_path = vox_files[0]
    print(f"Scanning VOX file: {input_vox_path}")

    parser = Vox200Parser(input_vox_path).parse()

    empty_layers = []
    for layer_name, voxels in parser.voxels_by_layer.items():
        if len(voxels) == 0:
            empty_layers.append(layer_name)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        if empty_layers:
            f.write("Empty Layers Detected:\n")
            for name in empty_layers:
                f.write(f"- {name}\n")
            print(f"Found {len(empty_layers)} empty layers.")
        else:
            f.write("No empty layers detected.\n")
            print("No empty layers detected.")

    print(f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
