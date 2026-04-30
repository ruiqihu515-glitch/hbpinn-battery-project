from pathlib import Path
from hbpinn_battery.data import find_mat_files

def main():
    """
    Check whether the NASA battery .mat files can be found.
    """
    project_root = Path(__file__).resolve().parents[1]

    workspace_root = project_root.parent

    data_dir = (
        workspace_root
        /"nasa_raw_data"
        /"5. Battery Data Set"
        /"extracted"
    )

    print("Project root:")
    print(project_root)

    print("\nNASA data directory:")
    print(data_dir)

    mat_files = find_mat_files(data_dir)

    print(f"\nFound {len(mat_files)} MATLAB .mat files.")

    if not mat_files:
        print("\nNo .mat files were found.")
        print("Please check whether the NASA files were extracted correctly.")
        return

    print("\nFirst files:")
    for file_path in mat_files[:10]:
        print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()