"""
Data utilities for the HB-PINN battery projects.

This moudle contains helper functions for finding and loading
NASA Li-ion battery data files.
"""

from pathlib import Path
import scipy.io

def find_mat_files(data_dir: str | Path) -> list[Path]:
    """
    Find all MATLAB .mat files inside a data directory.

    Parameters
    ----------
    data_dir:
        Path to the folder that contains NASA battery files.

    Returns
    -------
    list[Path]
        A sorted list of .mat file paths.
    """

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    mat_files = sorted(data_dir.rglob("*.mat"))

    return  mat_files


def load_mat_file(file_path: str | Path) -> dict:
    """
    Load a MATLAB .mat file.

    Parameters
    ----------
    file_path:
        Path to one NASA battery .mat file.

    Returns
    -------
    dict
        Raw dictionary loaded from the MATLAB file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"MAT file does not exist: {file_path}")

    mat_data = scipy.io.loadmat(file_path)

    return mat_data

def matlab_string_to_python(value) -> str:
    """
    Convert a MATLAB string loaded by scipy into a normal Python string.
    """
    if hasattr(value, "squeeze"):
        value = value.squeeze()

    if hasattr(value, "size") and value.size == 1:
        value = value.item()

    if isinstance(value, bytes):
        return value.decode("utf-8").strip()

    if isinstance(value, str):
        return value.strip()

    if hasattr(value, "ravel"):
        characters = [str(item) for item in value.ravel()]
        return "".join(characters).strip()

    return str(value).strip()