"""
Shared pytest fixtures and configuration for lidar_excavation tests.
"""

import pytest
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DEMO_LAS_PATH = PROJECT_ROOT / "demo_terrain.las"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_laspy: requires laspy to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_matplotlib: requires matplotlib to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_rasterio: requires rasterio for GeoTIFF tests"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on missing dependencies."""
    try:
        import laspy
        laspy_available = True
    except ImportError:
        laspy_available = False

    try:
        import matplotlib
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False

    try:
        import rasterio
        rasterio_available = True
    except ImportError:
        rasterio_available = False

    for item in items:
        if "requires_laspy" in item.keywords and not laspy_available:
            item.add_marker(pytest.mark.skip(reason="laspy not installed"))
        if "requires_matplotlib" in item.keywords and not matplotlib_available:
            item.add_marker(pytest.mark.skip(reason="matplotlib not installed"))
        if "requires_rasterio" in item.keywords and not rasterio_available:
            item.add_marker(pytest.mark.skip(reason="rasterio not installed"))


@pytest.fixture
def demo_las_path():
    """Path to demo LAS file."""
    if not DEMO_LAS_PATH.exists():
        pytest.skip("demo_terrain.las not found")
    return DEMO_LAS_PATH


@pytest.fixture
def sample_point_cloud():
    """Generate a small synthetic point cloud for fast tests."""
    from lidar_excavation.io.point_cloud import generate_sample_terrain

    return generate_sample_terrain(
        size=(20.0, 20.0),
        resolution=1.0,
        base_elevation=100.0,
        seed=42,
    )


@pytest.fixture
def sample_terrain(sample_point_cloud):
    """Generate a small terrain grid for fast tests."""
    from lidar_excavation.core.terrain import TerrainGrid

    return TerrainGrid.from_point_cloud(
        sample_point_cloud,
        resolution=1.0,
        ground_only=False,
    )


@pytest.fixture
def sample_polygon():
    """A simple test polygon in the center of the sample terrain."""
    from shapely.geometry import box

    return box(5, 5, 15, 15)


@pytest.fixture(scope="session")
def demo_point_cloud(request):
    """
    Load demo LAS file once per session.

    Marked as session-scoped for efficiency since loading is slow.
    """
    if not DEMO_LAS_PATH.exists():
        pytest.skip("demo_terrain.las not found")

    from lidar_excavation.io.point_cloud import PointCloudLoader

    return PointCloudLoader.load(DEMO_LAS_PATH)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    from click.testing import CliRunner

    return CliRunner()
