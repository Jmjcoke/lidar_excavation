"""
Streamlit Web Interface for LIDAR Excavation Analysis

Run with: streamlit run src/lidar_excavation/web.py
Or after install: lidar-excavation-web
"""

import subprocess
import sys
from pathlib import Path


def run():
    """Launch the Streamlit web app."""
    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    run()
