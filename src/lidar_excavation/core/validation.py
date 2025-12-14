"""
Input Validation Module

Provides validation functions and custom exceptions for the lidar_excavation package.
All validation functions provide clear, actionable error messages.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from shapely.geometry import Polygon


class ValidationError(ValueError):
    """Base exception for validation errors with user-friendly messages."""
    pass


class ResolutionError(ValidationError):
    """Invalid resolution value."""
    pass


class BoundsError(ValidationError):
    """Polygon/bounds don't intersect terrain."""
    pass


class EmptyResultError(ValidationError):
    """Calculation produced no results."""
    pass


class FilePermissionError(ValidationError):
    """Cannot write to specified path."""
    pass


def validate_resolution(resolution: float, context: str = "resolution") -> float:
    """
    Validate resolution is a positive number.

    Args:
        resolution: The resolution value to validate
        context: Description of what this resolution is for (used in error messages)

    Returns:
        The validated resolution as a float

    Raises:
        ResolutionError: If resolution is None, not a number, or <= 0
    """
    if resolution is None:
        raise ResolutionError(f"{context} cannot be None")

    if not isinstance(resolution, (int, float)):
        raise ResolutionError(
            f"{context} must be a number, got {type(resolution).__name__}"
        )

    if resolution <= 0:
        raise ResolutionError(
            f"{context} must be positive, got {resolution}. "
            "Typical values are 0.5-5.0 meters for terrain analysis."
        )

    return float(resolution)


def validate_slope_percent(slope: float, context: str = "slope") -> float:
    """
    Validate slope percentage is reasonable.

    Args:
        slope: The slope percentage to validate
        context: Description of what this slope is for (used in error messages)

    Returns:
        The validated slope as a float

    Raises:
        ValidationError: If slope is None, not a number, or negative
    """
    if slope is None:
        raise ValidationError(f"{context} cannot be None")

    if not isinstance(slope, (int, float)):
        raise ValidationError(
            f"{context} must be a number, got {type(slope).__name__}"
        )

    if slope < 0:
        raise ValidationError(
            f"{context} cannot be negative, got {slope}%. "
            "Use slope_direction to control which way the surface slopes."
        )

    if slope > 100:
        warnings.warn(
            f"{context} of {slope}% is unusually steep (>45 degrees). "
            "Verify this is intentional.",
            UserWarning,
            stacklevel=2
        )

    return float(slope)


def validate_soil_factor(factor: float, name: str) -> float:
    """
    Validate soil swell/shrink factor is reasonable.

    Args:
        factor: The soil factor to validate
        name: Name of the factor (e.g., "swell_factor", "shrink_factor")

    Returns:
        The validated factor as a float

    Raises:
        ValidationError: If factor is None, not a number, or <= 0
    """
    if factor is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(factor, (int, float)):
        raise ValidationError(
            f"{name} must be a number, got {type(factor).__name__}"
        )

    if factor <= 0:
        raise ValidationError(
            f"{name} must be positive, got {factor}. "
            "Typical swell factors are 1.1-1.5, shrink factors are 0.7-0.95."
        )

    if factor > 3.0 or factor < 0.3:
        warnings.warn(
            f"{name} of {factor} is outside typical range (0.3-3.0). "
            "Verify this is intentional.",
            UserWarning,
            stacklevel=2
        )

    return float(factor)


def validate_output_path(filepath: Union[str, Path], context: str = "output file") -> Path:
    """
    Validate output path is writable before attempting to write.

    Args:
        filepath: The path to validate
        context: Description of what will be written (used in error messages)

    Returns:
        The validated path as a Path object

    Raises:
        FilePermissionError: If directory doesn't exist or isn't writable
    """
    path = Path(filepath)
    parent = path.parent

    # Handle empty parent (current directory)
    if str(parent) == '.':
        parent = Path.cwd()

    # Check parent directory exists
    if not parent.exists():
        raise FilePermissionError(
            f"Cannot write {context}: directory '{parent}' does not exist. "
            "Create the directory first or specify a different path."
        )

    # Check parent directory is writable
    if not os.access(parent, os.W_OK):
        raise FilePermissionError(
            f"Cannot write {context}: no write permission for directory '{parent}'."
        )

    # Check if file exists and is writable (for overwrites)
    if path.exists() and not os.access(path, os.W_OK):
        raise FilePermissionError(
            f"Cannot overwrite {context}: file '{path}' exists but is not writable."
        )

    return path


def validate_polygon_terrain_intersection(
    polygon: 'Polygon',
    terrain_bounds: Tuple[float, float, float, float],
    terrain_name: str = "terrain"
) -> None:
    """
    Validate that polygon intersects with terrain bounds.

    Args:
        polygon: Shapely Polygon to check
        terrain_bounds: (min_x, min_y, max_x, max_y) of terrain
        terrain_name: Name of terrain for error messages

    Raises:
        BoundsError: If polygon doesn't intersect terrain
    """
    from shapely.geometry import box

    poly_bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    terrain_box = box(*terrain_bounds)

    if not polygon.intersects(terrain_box):
        raise BoundsError(
            f"The specified polygon does not intersect the {terrain_name}.\n"
            f"  Polygon bounds: X={poly_bounds[0]:.1f} to {poly_bounds[2]:.1f}, "
            f"Y={poly_bounds[1]:.1f} to {poly_bounds[3]:.1f}\n"
            f"  Terrain bounds: X={terrain_bounds[0]:.1f} to {terrain_bounds[2]:.1f}, "
            f"Y={terrain_bounds[1]:.1f} to {terrain_bounds[3]:.1f}\n"
            "Ensure your polygon coordinates match the terrain's coordinate system."
        )


def validate_grid_dimensions(
    rows: int,
    cols: int,
    bounds: Tuple[float, float, float, float],
    resolution: float
) -> None:
    """
    Validate that grid dimensions are valid.

    Args:
        rows: Number of rows
        cols: Number of columns
        bounds: (min_x, min_y, max_x, max_y) of the data
        resolution: Grid cell size

    Raises:
        ValidationError: If dimensions are invalid
    """
    if rows <= 0 or cols <= 0:
        min_x, min_y, max_x, max_y = bounds
        raise ValidationError(
            f"Invalid grid dimensions ({rows} rows x {cols} cols). "
            f"Check that bounds ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f}) "
            f"are valid with resolution {resolution}."
        )

    # Warn on very large grids
    total_cells = rows * cols
    if total_cells > 100_000_000:  # 100M cells
        warnings.warn(
            f"Creating very large grid ({rows}x{cols} = {total_cells:,} cells). "
            "Consider using a coarser resolution to reduce memory usage.",
            UserWarning,
            stacklevel=2
        )
