"""
Tests using real LAS file (demo_terrain.las).
"""

import pytest
import numpy as np


@pytest.mark.requires_laspy
class TestRealLASLoading:
    """Tests for loading and processing real LAS files."""

    def test_load_demo_las(self, demo_las_path):
        """Test loading demo_terrain.las file."""
        from lidar_excavation.io.point_cloud import PointCloudLoader

        pc = PointCloudLoader.load(demo_las_path)

        assert pc.num_points > 0
        assert pc.xyz.shape[1] == 3

    def test_demo_las_bounds(self, demo_point_cloud):
        """Test bounds of loaded LAS file are reasonable."""
        bounds_min, bounds_max = demo_point_cloud.bounds

        # Bounds should not be infinite or NaN
        assert np.all(np.isfinite(bounds_min))
        assert np.all(np.isfinite(bounds_max))
        # Max should be greater than min
        assert np.all(bounds_max > bounds_min)

    def test_demo_las_has_classification(self, demo_point_cloud):
        """Test that demo LAS has classification data."""
        # Most real LAS files have classification
        # This test documents what the demo file contains
        assert demo_point_cloud.classification is not None

    def test_demo_las_ground_filtering(self, demo_point_cloud):
        """Test filtering to ground points only."""
        if demo_point_cloud.classification is None:
            pytest.skip("No classification data in demo file")

        ground = demo_point_cloud.filter_by_classification([2])

        # Should have some ground points but fewer than total
        assert ground.num_points > 0
        assert ground.num_points <= demo_point_cloud.num_points
        assert all(ground.classification == 2)


@pytest.mark.requires_laspy
@pytest.mark.slow
class TestTerrainFromRealLAS:
    """Tests for creating terrain from real LAS data."""

    def test_terrain_from_demo_las(self, demo_point_cloud):
        """Test creating terrain grid from real LAS data."""
        from lidar_excavation.core.terrain import TerrainGrid

        terrain = TerrainGrid.from_point_cloud(
            demo_point_cloud,
            resolution=2.0,  # Coarser for speed
            ground_only=True,
        )

        assert terrain.rows > 0
        assert terrain.cols > 0

        stats = terrain.statistics()
        assert stats["valid_cells"] > 0

    def test_terrain_statistics_valid(self, demo_point_cloud):
        """Test terrain statistics from real data."""
        from lidar_excavation.core.terrain import TerrainGrid

        terrain = TerrainGrid.from_point_cloud(
            demo_point_cloud,
            resolution=2.0,
            ground_only=True,
        )

        stats = terrain.statistics()

        # Statistics should be reasonable
        assert stats["min_elevation"] < stats["max_elevation"]
        assert stats["mean_elevation"] >= stats["min_elevation"]
        assert stats["mean_elevation"] <= stats["max_elevation"]
        assert stats["std_elevation"] >= 0


@pytest.mark.requires_laspy
@pytest.mark.slow
class TestVolumeCalculationOnRealData:
    """Tests for volume calculation on real terrain."""

    def test_volume_calculation_on_demo_las(self, demo_point_cloud):
        """Test volume calculation on real terrain."""
        from lidar_excavation.core.terrain import TerrainGrid
        from lidar_excavation.core.volume import VolumeCalculator
        from shapely.geometry import box

        terrain = TerrainGrid.from_point_cloud(
            demo_point_cloud,
            resolution=2.0,
            ground_only=True,
        )

        bounds = terrain.bounds
        # Create a work area in the center of the terrain
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        work_area = box(center_x - 10, center_y - 10, center_x + 10, center_y + 10)

        calculator = VolumeCalculator(terrain)
        stats = terrain.statistics()
        result = calculator.calculate_flat(work_area, stats["mean_elevation"])

        assert result.total_area > 0
        assert result.cut_volume >= 0
        assert result.fill_volume >= 0

    def test_optimal_elevation_on_demo_las(self, demo_point_cloud):
        """Test finding optimal elevation on real terrain."""
        from lidar_excavation.core.terrain import TerrainGrid
        from lidar_excavation.core.volume import VolumeCalculator
        from shapely.geometry import box

        terrain = TerrainGrid.from_point_cloud(
            demo_point_cloud,
            resolution=2.0,
            ground_only=True,
        )

        bounds = terrain.bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        work_area = box(center_x - 10, center_y - 10, center_x + 10, center_y + 10)

        calculator = VolumeCalculator(terrain)
        optimal_elev, result = calculator.find_optimal_elevation(
            work_area, constraint="balance"
        )

        # Optimal elevation should be within terrain range
        stats = terrain.statistics()
        assert optimal_elev >= stats["min_elevation"]
        assert optimal_elev <= stats["max_elevation"]


@pytest.mark.requires_laspy
class TestChunkedLoading:
    """Test chunked loading for large files."""

    def test_chunked_loading_demo(self, demo_las_path):
        """Test chunked loading of demo file."""
        from lidar_excavation.io.point_cloud import PointCloudLoader

        chunks = list(PointCloudLoader.load_chunked(demo_las_path, chunk_size=1000))

        assert len(chunks) > 0
        total_points = sum(chunk.num_points for chunk in chunks)

        # Compare with full load
        full_pc = PointCloudLoader.load(demo_las_path)
        assert total_points == full_pc.num_points

    def test_chunks_have_valid_data(self, demo_las_path):
        """Test that each chunk has valid data."""
        from lidar_excavation.io.point_cloud import PointCloudLoader

        for chunk in PointCloudLoader.load_chunked(demo_las_path, chunk_size=500):
            assert chunk.num_points > 0
            assert chunk.xyz.shape[1] == 3
            bounds_min, bounds_max = chunk.bounds
            assert np.all(np.isfinite(bounds_min))
            assert np.all(np.isfinite(bounds_max))


@pytest.mark.requires_laspy
class TestCRSExtraction:
    """Test CRS extraction from LAS files."""

    def test_crs_extraction_demo(self, demo_point_cloud):
        """Test CRS extraction from demo file."""
        # The demo file may or may not have CRS - just verify no crash
        # and that the property exists
        crs = demo_point_cloud.crs
        # CRS can be None, a WKT string, or an EPSG code
        assert crs is None or isinstance(crs, str)
