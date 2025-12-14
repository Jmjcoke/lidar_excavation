"""
Tests for input validation module.
"""

import pytest
import warnings
from pathlib import Path

from lidar_excavation.core.validation import (
    ValidationError,
    ResolutionError,
    BoundsError,
    EmptyResultError,
    FilePermissionError,
    validate_resolution,
    validate_slope_percent,
    validate_soil_factor,
    validate_output_path,
    validate_polygon_terrain_intersection,
    validate_grid_dimensions,
)


class TestResolutionValidation:
    """Tests for resolution validation."""

    def test_valid_resolution_float(self):
        """Test valid float resolution."""
        assert validate_resolution(1.0) == 1.0
        assert validate_resolution(0.5) == 0.5
        assert validate_resolution(10.0) == 10.0

    def test_valid_resolution_int(self):
        """Test valid integer resolution (converted to float)."""
        assert validate_resolution(1) == 1.0
        assert validate_resolution(10) == 10.0

    def test_zero_resolution_raises(self):
        """Test that zero resolution raises ResolutionError."""
        with pytest.raises(ResolutionError, match="must be positive"):
            validate_resolution(0)

    def test_negative_resolution_raises(self):
        """Test that negative resolution raises ResolutionError."""
        with pytest.raises(ResolutionError, match="must be positive"):
            validate_resolution(-1.0)

    def test_none_resolution_raises(self):
        """Test that None resolution raises ResolutionError."""
        with pytest.raises(ResolutionError, match="cannot be None"):
            validate_resolution(None)

    def test_string_resolution_raises(self):
        """Test that string resolution raises ResolutionError."""
        with pytest.raises(ResolutionError, match="must be a number"):
            validate_resolution("1.0")

    def test_custom_context_in_message(self):
        """Test that custom context appears in error message."""
        with pytest.raises(ResolutionError, match="Grid spacing"):
            validate_resolution(-1, context="Grid spacing")


class TestSlopeValidation:
    """Tests for slope percentage validation."""

    def test_valid_slope(self):
        """Test valid slope values."""
        assert validate_slope_percent(0) == 0.0
        assert validate_slope_percent(2.0) == 2.0
        assert validate_slope_percent(50) == 50.0

    def test_negative_slope_raises(self):
        """Test that negative slope raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be negative"):
            validate_slope_percent(-5.0)

    def test_extreme_slope_warns(self):
        """Test that extreme slope (>100%) raises warning."""
        with pytest.warns(UserWarning, match="unusually steep"):
            validate_slope_percent(150.0)

    def test_none_slope_raises(self):
        """Test that None slope raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be None"):
            validate_slope_percent(None)


class TestSoilFactorValidation:
    """Tests for soil swell/shrink factor validation."""

    def test_valid_swell_factors(self):
        """Test typical valid swell factors."""
        assert validate_soil_factor(1.0, "swell") == 1.0
        assert validate_soil_factor(1.2, "swell") == 1.2
        assert validate_soil_factor(1.5, "swell") == 1.5

    def test_valid_shrink_factors(self):
        """Test typical valid shrink factors."""
        assert validate_soil_factor(0.9, "shrink") == 0.9
        assert validate_soil_factor(0.85, "shrink") == 0.85

    def test_zero_factor_raises(self):
        """Test that zero factor raises ValidationError."""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_soil_factor(0, "swell_factor")

    def test_negative_factor_raises(self):
        """Test that negative factor raises ValidationError."""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_soil_factor(-0.5, "shrink_factor")

    def test_extreme_high_factor_warns(self):
        """Test that extremely high factor raises warning."""
        with pytest.warns(UserWarning, match="outside typical range"):
            validate_soil_factor(5.0, "swell_factor")

    def test_extreme_low_factor_warns(self):
        """Test that extremely low factor raises warning."""
        with pytest.warns(UserWarning, match="outside typical range"):
            validate_soil_factor(0.1, "shrink_factor")


class TestOutputPathValidation:
    """Tests for output path validation."""

    def test_valid_path_in_existing_directory(self, tmp_path):
        """Test valid path in existing writable directory."""
        output_file = tmp_path / "output.json"
        result = validate_output_path(output_file)
        assert result == output_file

    def test_string_path_converted_to_path(self, tmp_path):
        """Test string path is converted to Path object."""
        output_file = str(tmp_path / "output.json")
        result = validate_output_path(output_file)
        assert isinstance(result, Path)

    def test_nonexistent_directory_raises(self, tmp_path):
        """Test that path in nonexistent directory raises error."""
        output_file = tmp_path / "nonexistent_dir" / "output.json"
        with pytest.raises(FilePermissionError, match="does not exist"):
            validate_output_path(output_file)

    def test_custom_context_in_message(self, tmp_path):
        """Test that custom context appears in error message."""
        output_file = tmp_path / "nonexistent" / "file.json"
        with pytest.raises(FilePermissionError, match="CSV output"):
            validate_output_path(output_file, context="CSV output")


class TestPolygonTerrainIntersection:
    """Tests for polygon-terrain intersection validation."""

    def test_intersecting_polygon_passes(self):
        """Test that intersecting polygon passes validation."""
        from shapely.geometry import box

        polygon = box(10, 10, 20, 20)
        terrain_bounds = (0, 0, 100, 100)
        # Should not raise
        validate_polygon_terrain_intersection(polygon, terrain_bounds)

    def test_non_intersecting_polygon_raises(self):
        """Test that non-intersecting polygon raises BoundsError."""
        from shapely.geometry import box

        polygon = box(200, 200, 300, 300)
        terrain_bounds = (0, 0, 100, 100)
        with pytest.raises(BoundsError, match="does not intersect"):
            validate_polygon_terrain_intersection(polygon, terrain_bounds)

    def test_partially_overlapping_polygon_passes(self):
        """Test that partially overlapping polygon passes."""
        from shapely.geometry import box

        polygon = box(90, 90, 150, 150)  # Overlaps corner
        terrain_bounds = (0, 0, 100, 100)
        # Should not raise
        validate_polygon_terrain_intersection(polygon, terrain_bounds)


class TestGridDimensionsValidation:
    """Tests for grid dimension validation."""

    def test_valid_dimensions_pass(self):
        """Test valid grid dimensions."""
        # Should not raise
        validate_grid_dimensions(100, 100, (0, 0, 100, 100), 1.0)

    def test_zero_rows_raises(self):
        """Test that zero rows raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid grid dimensions"):
            validate_grid_dimensions(0, 100, (0, 0, 100, 100), 1.0)

    def test_zero_cols_raises(self):
        """Test that zero cols raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid grid dimensions"):
            validate_grid_dimensions(100, 0, (0, 0, 100, 100), 1.0)

    def test_very_large_grid_warns(self):
        """Test that very large grid raises warning."""
        with pytest.warns(UserWarning, match="very large grid"):
            validate_grid_dimensions(15000, 15000, (0, 0, 15000, 15000), 1.0)


class TestValidationIntegration:
    """Integration tests for validation in actual components."""

    def test_terrain_resolution_validation(self, sample_point_cloud):
        """Test that TerrainGrid validates resolution."""
        from lidar_excavation.core.terrain import TerrainGrid

        with pytest.raises(ResolutionError):
            TerrainGrid.from_point_cloud(sample_point_cloud, resolution=0)

    def test_volume_calculator_soil_factor_validation(self, sample_terrain):
        """Test that VolumeCalculator validates soil factors."""
        from lidar_excavation.core.volume import VolumeCalculator

        with pytest.raises(ValidationError):
            VolumeCalculator(sample_terrain, swell_factor=-1.0)

    def test_volume_calculator_polygon_validation(self, sample_terrain):
        """Test that VolumeCalculator validates polygon intersection."""
        from lidar_excavation.core.volume import VolumeCalculator
        from shapely.geometry import box

        calculator = VolumeCalculator(sample_terrain)
        far_away_polygon = box(1000, 1000, 2000, 2000)

        with pytest.raises(BoundsError):
            calculator.calculate_flat(far_away_polygon, target_elevation=100)
