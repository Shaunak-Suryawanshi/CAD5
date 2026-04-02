import importlib.util
from pathlib import Path

def test_import():
    script = Path("3d_to_2d_cad.py")
    assert script.exists()