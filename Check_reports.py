import os
import sys

BASE_DIR = os.path.dirname(__file__)
REPORTS = [
  "CompareReport_Character.txt",
  "DebugNameMap_Character.txt",
  "FinalMapping_Character.csv",
  "FinalMapping_Character.txt",
  "EmptyLayerReport.txt",
  "LayerMapping_Character.txt",
  "name_overrides_heur_suggested_Character.json",
  "name_overrides_highconf_Character.json",
  "name_overrides_suggested_Character.json",
  "name_overrides_suggested_Character.txt",
  "RenamingManifest_Character.json",
  "RenamingReport_Character.txt",
  "VoxelOverrideReport_Character.txt"
]

def main():
    reports_dir = os.path.join(BASE_DIR, "reports")
    found = []
    missing = []
    for r in REPORTS:
        path = os.path.join(reports_dir, r) if not os.path.isabs(r) else r
        if os.path.exists(path):
            found.append(path)
        else:
            missing.append(path)
    print("Reports directory:", reports_dir)
    print("\nFound reports:")
    for p in found:
        print(" ", os.path.relpath(p, BASE_DIR))
    print("\nMissing reports:")
    for p in missing:
        print(" ", os.path.relpath(p, BASE_DIR))
    if missing:
        print("\nTo recreate missing reports, run the producer scripts listed in the README or the guidance above.")
        # Exit non-zero so callers (like the .bat) can detect missing reports
        sys.exit(1)
    # All present
    sys.exit(0)

if __name__ == '__main__':
    main()