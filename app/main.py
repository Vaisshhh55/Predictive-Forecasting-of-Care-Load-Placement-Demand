from pathlib import Path
import sys


project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.dashboard import render_dashboard


if __name__ == "__main__":
    render_dashboard()
