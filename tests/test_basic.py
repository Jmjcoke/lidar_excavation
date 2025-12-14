"""
Basic tests for the lidar_excavation package.

Run with: pytest tests/
"""

import numpy as np
import pytest
from shapely.geometry import box


class TestPointCloud:
    """Tests for point cloud loading and manipulation."""

    def test_generate_sample_terrain(self):
        """Test synthetic terrain generation."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain

        pc = generate_sample_terrain(
            size=(50.0, 50.0),
            resolution=1.0,
            base_elevation=100.0,
        )

        assert pc.num_points > 0
        assert pc.xyz.shape[1] == 3

        bounds_min, bounds_max = pc.bounds
        assert bounds_min[0] >= 0
        assert bounds_max[0] <= 50
        assert bounds_min[1] >= 0
        assert bounds_max[1] <= 50

    def test_point_cloud_filtering(self):
        """Test point cloud spatial filtering."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain

        pc = generate_sample_terrain(size=(100.0, 100.0), resolution=1.0)

        # Filter to subset
        filtered = pc.filter_by_bounds(min_x=25, max_x=75, min_y=25, max_y=75)

        assert filtered.num_points < pc.num_points
        assert all(filtered.x >= 25)
        assert all(filtered.x <= 75)

    def test_point_cloud_subsample(self):
        """Test point cloud subsampling."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain

        pc = generate_sample_terrain(size=(100.0, 100.0), resolution=1.0)
        subsampled = pc.subsample(factor=10)

        assert subsampled.num_points == pc.num_points // 10


class TestTerrainGrid:
    """Tests for terrain grid generation and manipulation."""

    def test_terrain_from_point_cloud(self):
        """Test terrain grid creation from point cloud."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain
        from lidar_excavation.core.terrain import TerrainGrid

        pc = generate_sample_terrain(size=(50.0, 50.0), resolution=1.0)
        terrain = TerrainGrid.from_point_cloud(pc, resolution=2.0)

        assert terrain.rows > 0
        assert terrain.cols > 0
        assert terrain.resolution == 2.0

    def test_terrain_coordinate_conversion(self):
        """Test coordinate to cell conversion."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain
        from lidar_excavation.core.terrain import TerrainGrid

        pc = generate_sample_terrain(size=(100.0, 100.0), resolution=1.0)
        terrain = TerrainGrid.from_point_cloud(pc, resolution=5.0)

        # Test round-trip conversion
        row, col = terrain.coord_to_cell(50.0, 50.0)
        x, y = terrain.cell_to_coord(row, col)

        assert abs(x - 50.0) < terrain.resolution
        assert abs(y - 50.0) < terrain.resolution

    def test_terrain_statistics(self):
        """Test terrain statistics calculation."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain
        from lidar_excavation.core.terrain import TerrainGrid

        pc = generate_sample_terrain(
            size=(50.0, 50.0),
            resolution=1.0,
            base_elevation=100.0,
        )
        terrain = TerrainGrid.from_point_cloud(pc, resolution=1.0)
        stats = terrain.statistics()

        assert "min_elevation" in stats
        assert "max_elevation" in stats
        assert stats["min_elevation"] < stats["max_elevation"]
        assert stats["valid_cells"] > 0


class TestVolumeCalculator:
    """Tests for cut/fill volume calculations."""

    def test_flat_grade_calculation(self):
        """Test flat grade volume calculation."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain
        from lidar_excavation.core.terrain import TerrainGrid
        from lidar_excavation.core.volume import VolumeCalculator

        # Create terrain with known elevation
        pc = generate_sample_terrain(
            size=(100.0, 100.0),
            resolution=1.0,
            base_elevation=100.0,
            hill_height=0.0,  # Flat terrain
            noise_scale=0.0,  # No noise
        )
        terrain = TerrainGrid.from_point_cloud(pc, resolution=1.0)

        # Calculate for a rectangle
        work_area = box(25, 25, 75, 75)  # 50x50 area
        calculator = VolumeCalculator(terrain)

        # Target above existing: should be all fill
        result_fill = calculator.calculate_flat(work_area, target_elevation=105.0)
        assert result_fill.fill_volume > 0
        assert result_fill.cut_volume < result_fill.fill_volume

        # Target below existing: should be all cut
        result_cut = calculator.calculate_flat(work_area, target_elevation=95.0)
        assert result_cut.cut_volume > 0
        assert result_cut.fill_volume < result_cut.cut_volume

    def test_balance_elevation(self):
        """Test finding balance elevation."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain
        from lidar_excavation.core.terrain import TerrainGrid
        from lidar_excavation.core.volume import VolumeCalculator

        pc = generate_sample_terrain(size=(50.0, 50.0), resolution=1.0)
        terrain = TerrainGrid.from_point_cloud(pc, resolution=1.0)

        work_area = box(10, 10, 40, 40)
        calculator = VolumeCalculator(terrain)

        result = calculator.calculate_flat(work_area, target_elevation=100.0)

        # Balance elevation should exist
        assert result.balance_elevation is not None

    def test_optimal_elevation(self):
        """Test finding optimal elevation."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain
        from lidar_excavation.core.terrain import TerrainGrid
        from lidar_excavation.core.volume import VolumeCalculator

        pc = generate_sample_terrain(size=(50.0, 50.0), resolution=1.0)
        terrain = TerrainGrid.from_point_cloud(pc, resolution=1.0)

        work_area = box(10, 10, 40, 40)
        calculator = VolumeCalculator(terrain)

        # Find optimal for balanced earthwork
        optimal_elev, result = calculator.find_optimal_elevation(
            work_area,
            constraint="balance"
        )

        # At optimal elevation, cut and fill should be close
        assert abs(result.cut_volume - result.fill_volume) < abs(result.cut_volume) * 0.1

    def test_sloped_grade_calculation(self):
        """Test sloped grade volume calculation."""
        from lidar_excavation.io.point_cloud import generate_sample_terrain
        from lidar_excavation.core.terrain import TerrainGrid
        from lidar_excavation.core.volume import VolumeCalculator

        pc = generate_sample_terrain(
            size=(100.0, 100.0),
            resolution=1.0,
            base_elevation=100.0,
            hill_height=0.0,
            noise_scale=0.0,
        )
        terrain = TerrainGrid.from_point_cloud(pc, resolution=1.0)

        work_area = box(25, 25, 75, 75)
        calculator = VolumeCalculator(terrain)

        result = calculator.calculate_sloped(
            work_area,
            base_elevation=100.0,
            slope_percent=2.0,
            slope_direction=180.0,  # Slope south
        )

        # With sloped target on flat terrain, should have both cut and fill
        assert result.cut_volume > 0 or result.fill_volume > 0
        assert result.total_area > 0


class TestGradingDesign:
    """Tests for grading design classes."""

    def test_flat_grade(self):
        """Test flat grade design."""
        from lidar_excavation.analysis.grading import FlatGrade, create_rectangle

        polygon = create_rectangle(50, 50, 20, 20)
        grade = FlatGrade(
            name="test_pad",
            polygon=polygon,
            elevation=100.0,
        )

        assert grade.target_elevation_at(50, 50) == 100.0
        assert grade.target_elevation_at(40, 60) == 100.0

    def test_sloped_grade(self):
        """Test sloped grade design."""
        from lidar_excavation.analysis.grading import SlopedGrade, create_rectangle

        polygon = create_rectangle(50, 50, 20, 20)
        grade = SlopedGrade(
            name="test_parking",
            polygon=polygon,
            base_elevation=100.0,
            slope_percent=2.0,
            slope_direction=180.0,  # Slope south
        )

        # Elevation should decrease going south
        elev_north = grade.target_elevation_at(50, 60)
        elev_south = grade.target_elevation_at(50, 40)

        assert elev_north > elev_south

    def test_create_rectangle(self):
        """Test rectangle creation utility."""
        from lidar_excavation.analysis.grading import create_rectangle

        rect = create_rectangle(100, 100, 50, 30)

        assert rect.area == pytest.approx(50 * 30)
        assert rect.centroid.x == pytest.approx(100)
        assert rect.centroid.y == pytest.approx(100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
