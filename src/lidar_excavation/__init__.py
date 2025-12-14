"""
LIDAR Excavation Analysis Tool

A Python library for analyzing LIDAR point clouds to calculate
excavation requirements for construction projects.
"""

__version__ = "0.1.0"

from .core.terrain import TerrainGrid
from .core.volume import VolumeCalculator, EarthworkResult
from .io.point_cloud import PointCloudLoader
from .analysis.grading import GradingDesign, FlatGrade, SlopedGrade

__all__ = [
    "TerrainGrid",
    "VolumeCalculator",
    "EarthworkResult",
    "PointCloudLoader",
    "GradingDesign",
    "FlatGrade",
    "SlopedGrade",
]
