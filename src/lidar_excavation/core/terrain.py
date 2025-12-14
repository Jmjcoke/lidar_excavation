"""
Terrain Grid Module

Converts point cloud data to a regular elevation grid (DEM)
for analysis and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Callable
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from ..io.point_cloud import PointCloud, PointCloudLoader


class InterpolationMethod(Enum):
    """Methods for interpolating elevation values within grid cells."""
    NEAREST = "nearest"      # Nearest point
    MEAN = "mean"            # Mean of all points in cell
    MIN = "min"              # Minimum (good for ground)
    MAX = "max"              # Maximum
    IDW = "idw"              # Inverse distance weighting
    LINEAR = "linear"        # Linear interpolation (scipy)
    CUBIC = "cubic"          # Cubic interpolation (scipy)


@dataclass
class TerrainGrid:
    """
    Regular grid representation of terrain elevation.

    Attributes:
        elevations: 2D array of elevation values [rows, cols]
        origin: (x, y) coordinates of grid origin (lower-left corner)
        resolution: Grid cell size in coordinate units
        nodata: Value used for cells with no data
        crs: Coordinate reference system
    """
    elevations: np.ndarray
    origin: Tuple[float, float]
    resolution: float
    nodata: float = -9999.0
    crs: Optional[str] = None

    # Cached derived data
    _slope: Optional[np.ndarray] = field(default=None, repr=False)
    _aspect: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid dimensions (rows, cols)."""
        return self.elevations.shape

    @property
    def rows(self) -> int:
        return self.elevations.shape[0]

    @property
    def cols(self) -> int:
        return self.elevations.shape[1]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Spatial bounds (min_x, min_y, max_x, max_y)."""
        min_x, min_y = self.origin
        max_x = min_x + self.cols * self.resolution
        max_y = min_y + self.rows * self.resolution
        return (min_x, min_y, max_x, max_y)

    @property
    def x_coords(self) -> np.ndarray:
        """X coordinates of cell centers."""
        min_x = self.origin[0] + self.resolution / 2
        return np.arange(min_x, min_x + self.cols * self.resolution, self.resolution)

    @property
    def y_coords(self) -> np.ndarray:
        """Y coordinates of cell centers."""
        min_y = self.origin[1] + self.resolution / 2
        return np.arange(min_y, min_y + self.rows * self.resolution, self.resolution)

    def cell_to_coord(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center)."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return (x, y)

    def coord_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return (row, col)

    def get_elevation(self, x: float, y: float) -> float:
        """Get elevation at world coordinates (nearest cell)."""
        row, col = self.coord_to_cell(x, y)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.elevations[row, col]
        return self.nodata

    def get_elevation_interpolated(self, x: float, y: float) -> float:
        """Get elevation at world coordinates using bilinear interpolation."""
        # Convert to continuous cell coordinates
        col_f = (x - self.origin[0]) / self.resolution - 0.5
        row_f = (y - self.origin[1]) / self.resolution - 0.5

        # Get surrounding cell indices
        col0 = int(np.floor(col_f))
        col1 = col0 + 1
        row0 = int(np.floor(row_f))
        row1 = row0 + 1

        # Check bounds
        if col0 < 0 or col1 >= self.cols or row0 < 0 or row1 >= self.rows:
            return self.nodata

        # Bilinear interpolation weights
        wx = col_f - col0
        wy = row_f - row0

        z00 = self.elevations[row0, col0]
        z01 = self.elevations[row0, col1]
        z10 = self.elevations[row1, col0]
        z11 = self.elevations[row1, col1]

        # Check for nodata
        if any(z == self.nodata for z in [z00, z01, z10, z11]):
            return self.nodata

        # Interpolate
        z0 = z00 * (1 - wx) + z01 * wx
        z1 = z10 * (1 - wx) + z11 * wx
        return z0 * (1 - wy) + z1 * wy

    @property
    def slope(self) -> np.ndarray:
        """
        Calculate slope in degrees.

        Uses Horn's method for slope calculation.
        """
        if self._slope is not None:
            return self._slope

        # Pad edges for gradient calculation
        z = np.pad(self.elevations, 1, mode='edge')

        # Horn's method: weighted gradients
        dz_dx = (
            (z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) -
            (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])
        ) / (8 * self.resolution)

        dz_dy = (
            (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:]) -
            (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])
        ) / (8 * self.resolution)

        # Slope in degrees
        self._slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

        # Mask nodata areas
        nodata_mask = self.elevations == self.nodata
        self._slope[nodata_mask] = self.nodata

        return self._slope

    @property
    def aspect(self) -> np.ndarray:
        """
        Calculate aspect (downslope direction) in degrees.

        0 = North, 90 = East, 180 = South, 270 = West
        """
        if self._aspect is not None:
            return self._aspect

        z = np.pad(self.elevations, 1, mode='edge')

        dz_dx = (
            (z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) -
            (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])
        ) / (8 * self.resolution)

        dz_dy = (
            (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:]) -
            (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])
        ) / (8 * self.resolution)

        # Aspect in degrees (0 = North, clockwise)
        self._aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))
        self._aspect = (self._aspect + 360) % 360

        # Mask nodata and flat areas
        nodata_mask = self.elevations == self.nodata
        flat_mask = (dz_dx == 0) & (dz_dy == 0)
        self._aspect[nodata_mask | flat_mask] = self.nodata

        return self._aspect

    def fill_voids(self, max_iterations: int = 100) -> TerrainGrid:
        """
        Fill nodata cells by interpolating from neighbors.

        Returns a new TerrainGrid with voids filled.
        """
        elevations = self.elevations.copy()
        nodata_mask = elevations == self.nodata

        for _ in range(max_iterations):
            if not np.any(nodata_mask):
                break

            # For each nodata cell, average valid neighbors
            new_elevations = elevations.copy()

            for row in range(self.rows):
                for col in range(self.cols):
                    if not nodata_mask[row, col]:
                        continue

                    # Get valid neighbors
                    neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                                val = elevations[nr, nc]
                                if val != self.nodata:
                                    neighbors.append(val)

                    if neighbors:
                        new_elevations[row, col] = np.mean(neighbors)
                        nodata_mask[row, col] = False

            elevations = new_elevations

        return TerrainGrid(
            elevations=elevations,
            origin=self.origin,
            resolution=self.resolution,
            nodata=self.nodata,
            crs=self.crs,
        )

    def smooth(self, sigma: float = 1.0) -> TerrainGrid:
        """Apply Gaussian smoothing to the terrain."""
        # Preserve nodata
        nodata_mask = self.elevations == self.nodata
        elevations = self.elevations.copy()
        elevations[nodata_mask] = np.nan

        # Apply filter
        smoothed = gaussian_filter(
            np.nan_to_num(elevations, nan=np.nanmean(elevations)),
            sigma=sigma
        )
        smoothed[nodata_mask] = self.nodata

        return TerrainGrid(
            elevations=smoothed,
            origin=self.origin,
            resolution=self.resolution,
            nodata=self.nodata,
            crs=self.crs,
        )

    def clip_to_bounds(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float
    ) -> TerrainGrid:
        """Extract a subset of the terrain grid."""
        # Convert bounds to cell indices
        min_row, min_col = self.coord_to_cell(min_x, min_y)
        max_row, max_col = self.coord_to_cell(max_x, max_y)

        # Clamp to valid range
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.rows, max_row + 1)
        max_col = min(self.cols, max_col + 1)

        # Extract subset
        elevations = self.elevations[min_row:max_row, min_col:max_col].copy()

        # New origin
        new_origin = self.cell_to_coord(min_row, min_col)
        new_origin = (new_origin[0] - self.resolution / 2,
                      new_origin[1] - self.resolution / 2)

        return TerrainGrid(
            elevations=elevations,
            origin=new_origin,
            resolution=self.resolution,
            nodata=self.nodata,
            crs=self.crs,
        )

    def statistics(self) -> dict:
        """Calculate basic statistics for the terrain."""
        valid = self.elevations[self.elevations != self.nodata]

        if len(valid) == 0:
            return {"error": "No valid elevation data"}

        return {
            "min_elevation": float(np.min(valid)),
            "max_elevation": float(np.max(valid)),
            "mean_elevation": float(np.mean(valid)),
            "std_elevation": float(np.std(valid)),
            "elevation_range": float(np.max(valid) - np.min(valid)),
            "valid_cells": int(len(valid)),
            "total_cells": int(self.elevations.size),
            "nodata_cells": int(np.sum(self.elevations == self.nodata)),
            "mean_slope": float(np.mean(self.slope[self.slope != self.nodata])),
        }

    @classmethod
    def from_point_cloud(
        cls,
        point_cloud: PointCloud,
        resolution: float = 1.0,
        method: InterpolationMethod = InterpolationMethod.MEAN,
        ground_only: bool = True,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> TerrainGrid:
        """
        Create terrain grid from point cloud.

        Args:
            point_cloud: Input point cloud
            resolution: Grid cell size in coordinate units
            method: Interpolation method for cell elevations
            ground_only: If True, only use ground-classified points (class 2)
            bounds: Optional (min_x, min_y, max_x, max_y) to limit extent

        Returns:
            TerrainGrid instance

        Raises:
            ResolutionError: If resolution is not positive
            ValidationError: If grid dimensions are invalid
        """
        from .validation import validate_resolution, validate_grid_dimensions

        # Validate resolution
        resolution = validate_resolution(resolution, "Grid resolution")

        # Filter to ground points if requested
        if ground_only and point_cloud.classification is not None:
            pc = point_cloud.filter_by_classification([PointCloudLoader.CLASS_GROUND])
        else:
            pc = point_cloud

        if pc.num_points == 0:
            raise ValueError("No points available after filtering")

        # Determine grid extent
        if bounds:
            min_x, min_y, max_x, max_y = bounds
        else:
            pc_min, pc_max = pc.bounds
            min_x, min_y = pc_min[0], pc_min[1]
            max_x, max_y = pc_max[0], pc_max[1]

        # Calculate grid dimensions
        cols = int(np.ceil((max_x - min_x) / resolution))
        rows = int(np.ceil((max_y - min_y) / resolution))

        # Validate grid dimensions
        validate_grid_dimensions(rows, cols, (min_x, min_y, max_x, max_y), resolution)

        origin = (min_x, min_y)

        # Create elevation grid
        if method in [InterpolationMethod.LINEAR, InterpolationMethod.CUBIC]:
            # Use scipy's griddata for interpolation
            elevations = cls._interpolate_scipy(
                pc.xyz, origin, resolution, rows, cols, method
            )
        else:
            # Use binning methods
            elevations = cls._interpolate_binned(
                pc.xyz, origin, resolution, rows, cols, method
            )

        return cls(
            elevations=elevations,
            origin=origin,
            resolution=resolution,
            crs=point_cloud.crs,
        )

    @staticmethod
    def _interpolate_binned(
        xyz: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        rows: int,
        cols: int,
        method: InterpolationMethod,
    ) -> np.ndarray:
        """Bin points into cells and compute elevation."""
        nodata = -9999.0
        elevations = np.full((rows, cols), nodata)

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Compute cell indices for all points
        col_indices = ((x - origin[0]) / resolution).astype(int)
        row_indices = ((y - origin[1]) / resolution).astype(int)

        # Filter valid indices
        valid = (
            (col_indices >= 0) & (col_indices < cols) &
            (row_indices >= 0) & (row_indices < rows)
        )
        col_indices = col_indices[valid]
        row_indices = row_indices[valid]
        z_valid = z[valid]

        # Group points by cell
        cell_indices = row_indices * cols + col_indices

        # Process each unique cell
        for cell_idx in np.unique(cell_indices):
            row = cell_idx // cols
            col = cell_idx % cols
            mask = cell_indices == cell_idx
            cell_z = z_valid[mask]

            if len(cell_z) == 0:
                continue

            if method == InterpolationMethod.MEAN:
                elevations[row, col] = np.mean(cell_z)
            elif method == InterpolationMethod.MIN:
                elevations[row, col] = np.min(cell_z)
            elif method == InterpolationMethod.MAX:
                elevations[row, col] = np.max(cell_z)
            elif method == InterpolationMethod.NEAREST:
                # Use point closest to cell center
                cell_center_x = origin[0] + (col + 0.5) * resolution
                cell_center_y = origin[1] + (row + 0.5) * resolution
                points_mask = mask
                distances = (
                    (x[valid][points_mask] - cell_center_x)**2 +
                    (y[valid][points_mask] - cell_center_y)**2
                )
                elevations[row, col] = cell_z[np.argmin(distances)]

        return elevations

    @staticmethod
    def _interpolate_scipy(
        xyz: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        rows: int,
        cols: int,
        method: InterpolationMethod,
    ) -> np.ndarray:
        """Use scipy griddata for interpolation."""
        nodata = -9999.0

        # Create grid coordinates
        x_coords = np.arange(origin[0] + resolution/2,
                            origin[0] + cols * resolution,
                            resolution)
        y_coords = np.arange(origin[1] + resolution/2,
                            origin[1] + rows * resolution,
                            resolution)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Interpolate
        method_str = 'linear' if method == InterpolationMethod.LINEAR else 'cubic'
        elevations = griddata(
            xyz[:, :2],  # xy points
            xyz[:, 2],   # z values
            (xx, yy),
            method=method_str,
            fill_value=nodata
        )

        return elevations.astype(np.float64)

    def to_geotiff(self, filepath: str) -> None:
        """Export terrain grid to GeoTIFF (requires rasterio)."""
        from .validation import validate_output_path

        # Validate output path before importing rasterio
        filepath = str(validate_output_path(filepath, "GeoTIFF output"))

        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF export. "
                "Install with: pip install rasterio"
            )

        transform = from_origin(
            self.origin[0],
            self.origin[1] + self.rows * self.resolution,
            self.resolution,
            self.resolution
        )

        # Flip vertically (GeoTIFF convention)
        data = np.flipud(self.elevations)

        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=self.rows,
            width=self.cols,
            count=1,
            dtype=data.dtype,
            crs=self.crs,
            transform=transform,
            nodata=self.nodata,
        ) as dst:
            dst.write(data, 1)
