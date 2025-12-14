"""
Volume Calculator Module

Calculates cut and fill volumes for excavation planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .terrain import TerrainGrid
    from ..analysis.grading import GradingDesign
    from shapely.geometry import Polygon


@dataclass
class CellDetail:
    """Detailed cut/fill information for a single grid cell."""
    row: int
    col: int
    x: float
    y: float
    existing_elevation: float
    target_elevation: float
    cut_depth: float  # Positive if cutting
    fill_depth: float  # Positive if filling
    cut_volume: float
    fill_volume: float
    area: float  # Actual area within polygon (may be partial cell)


@dataclass
class EarthworkResult:
    """
    Complete earthwork analysis results.

    All volumes are in cubic units (m³ if coordinates are in meters).
    """
    # Summary volumes
    cut_volume: float
    fill_volume: float
    net_volume: float  # Positive = net export, Negative = net import

    # Areas
    cut_area: float
    fill_area: float
    total_area: float

    # Depth statistics
    max_cut_depth: float
    max_fill_depth: float
    avg_cut_depth: float
    avg_fill_depth: float

    # Balance point (elevation where cut = fill)
    balance_elevation: Optional[float]

    # Detailed per-cell results
    cell_details: List[CellDetail]

    # Adjusted volumes (accounting for soil factors)
    adjusted_cut_volume: Optional[float] = None
    adjusted_fill_volume: Optional[float] = None

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 50,
            "EARTHWORK ANALYSIS SUMMARY",
            "=" * 50,
            f"Total Area:        {self.total_area:,.1f} sq units",
            f"",
            f"CUT (Excavation):",
            f"  Volume:          {self.cut_volume:,.1f} cubic units",
            f"  Area:            {self.cut_area:,.1f} sq units",
            f"  Max Depth:       {self.max_cut_depth:.2f} units",
            f"  Avg Depth:       {self.avg_cut_depth:.2f} units",
            f"",
            f"FILL (Embankment):",
            f"  Volume:          {self.fill_volume:,.1f} cubic units",
            f"  Area:            {self.fill_area:,.1f} sq units",
            f"  Max Depth:       {self.max_fill_depth:.2f} units",
            f"  Avg Depth:       {self.avg_fill_depth:.2f} units",
            f"",
            f"NET VOLUME:        {self.net_volume:,.1f} cubic units",
            f"  {'(Export required)' if self.net_volume > 0 else '(Import required)'}",
        ]

        if self.balance_elevation is not None:
            lines.append(f"")
            lines.append(f"Balance Elevation: {self.balance_elevation:.2f} units")
            lines.append(f"  (Elevation where cut = fill)")

        if self.adjusted_cut_volume is not None:
            lines.extend([
                f"",
                f"ADJUSTED VOLUMES (with soil factors):",
                f"  Cut:             {self.adjusted_cut_volume:,.1f} cubic units",
                f"  Fill:            {self.adjusted_fill_volume:,.1f} cubic units",
            ])

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "cut_volume": self.cut_volume,
            "fill_volume": self.fill_volume,
            "net_volume": self.net_volume,
            "cut_area": self.cut_area,
            "fill_area": self.fill_area,
            "total_area": self.total_area,
            "max_cut_depth": self.max_cut_depth,
            "max_fill_depth": self.max_fill_depth,
            "avg_cut_depth": self.avg_cut_depth,
            "avg_fill_depth": self.avg_fill_depth,
            "balance_elevation": self.balance_elevation,
            "adjusted_cut_volume": self.adjusted_cut_volume,
            "adjusted_fill_volume": self.adjusted_fill_volume,
        }


class VolumeCalculator:
    """
    Calculates cut/fill volumes for grading designs.

    Supports:
    - Simple flat grading (constant elevation)
    - Sloped grading (planar surface with slope)
    - Complex polygon boundaries
    - Soil swell/shrink factors
    """

    # Typical soil factors (excavated volume vs compacted volume)
    SOIL_SWELL_FACTORS = {
        "sand": 1.12,      # Sand expands ~12% when excavated
        "clay": 1.30,      # Clay expands ~30%
        "rock": 1.50,      # Rock expands ~50%
        "topsoil": 1.25,   # Topsoil expands ~25%
        "gravel": 1.15,    # Gravel expands ~15%
    }

    SOIL_SHRINK_FACTORS = {
        "sand": 0.95,      # Sand compacts to ~95% of loose volume
        "clay": 0.85,      # Clay compacts to ~85%
        "rock": 0.70,      # Crushed rock compacts significantly
        "topsoil": 0.90,   # Topsoil compacts to ~90%
        "gravel": 0.92,    # Gravel compacts to ~92%
    }

    def __init__(
        self,
        terrain: 'TerrainGrid',
        swell_factor: float = 1.0,
        shrink_factor: float = 1.0,
    ):
        """
        Initialize calculator with terrain data.

        Args:
            terrain: TerrainGrid representing existing ground
            swell_factor: Multiplier for cut volumes (excavated soil expands)
            shrink_factor: Multiplier for fill volumes (placed soil compacts)

        Raises:
            ValidationError: If soil factors are invalid
        """
        from .validation import validate_soil_factor

        self.terrain = terrain
        self.swell_factor = validate_soil_factor(swell_factor, "swell_factor")
        self.shrink_factor = validate_soil_factor(shrink_factor, "shrink_factor")

    def calculate_flat(
        self,
        polygon: 'Polygon',
        target_elevation: float,
    ) -> EarthworkResult:
        """
        Calculate cut/fill for a flat grading design.

        Args:
            polygon: Shapely Polygon defining the work area
            target_elevation: Desired finish elevation

        Returns:
            EarthworkResult with volumes and details

        Raises:
            BoundsError: If polygon doesn't intersect terrain
            EmptyResultError: If no valid cells found within polygon
        """
        from shapely.geometry import box
        from .validation import validate_polygon_terrain_intersection, EmptyResultError

        # Validate polygon intersects terrain
        validate_polygon_terrain_intersection(
            polygon, self.terrain.bounds, "terrain grid"
        )

        cell_details = []
        total_cut = 0.0
        total_fill = 0.0
        cut_area = 0.0
        fill_area = 0.0
        cut_depths = []
        fill_depths = []

        resolution = self.terrain.resolution
        cell_area = resolution * resolution

        # Iterate over grid cells that might intersect polygon
        min_x, min_y, max_x, max_y = polygon.bounds

        # Convert to cell indices
        min_row, min_col = self.terrain.coord_to_cell(min_x, min_y)
        max_row, max_col = self.terrain.coord_to_cell(max_x, max_y)

        # Clamp to valid range
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.terrain.rows - 1, max_row)
        max_col = min(self.terrain.cols - 1, max_col)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                # Get cell center and bounds
                cx, cy = self.terrain.cell_to_coord(row, col)
                cell_box = box(
                    cx - resolution / 2,
                    cy - resolution / 2,
                    cx + resolution / 2,
                    cy + resolution / 2,
                )

                # Check intersection with polygon
                if not polygon.intersects(cell_box):
                    continue

                # Calculate intersection area
                intersection = polygon.intersection(cell_box)
                area = intersection.area

                if area < 1e-10:
                    continue

                # Get existing elevation
                existing = self.terrain.elevations[row, col]

                if existing == self.terrain.nodata:
                    continue

                # Calculate cut/fill
                delta = existing - target_elevation

                cut_depth = max(0.0, delta)
                fill_depth = max(0.0, -delta)
                cut_vol = cut_depth * area
                fill_vol = fill_depth * area

                total_cut += cut_vol
                total_fill += fill_vol

                if cut_depth > 0:
                    cut_area += area
                    cut_depths.append(cut_depth)
                if fill_depth > 0:
                    fill_area += area
                    fill_depths.append(fill_depth)

                cell_details.append(CellDetail(
                    row=row,
                    col=col,
                    x=cx,
                    y=cy,
                    existing_elevation=existing,
                    target_elevation=target_elevation,
                    cut_depth=cut_depth,
                    fill_depth=fill_depth,
                    cut_volume=cut_vol,
                    fill_volume=fill_vol,
                    area=area,
                ))

        # Check for empty results
        if not cell_details:
            raise EmptyResultError(
                "No valid terrain cells found within the polygon. "
                "This may occur if:\n"
                "  - The polygon is entirely in a nodata region\n"
                "  - The polygon is too small for the grid resolution\n"
                f"  - Grid resolution: {self.terrain.resolution}, "
                f"Polygon area: {polygon.area:.2f}"
            )

        # Calculate statistics
        total_area = cut_area + fill_area
        max_cut = max(cut_depths) if cut_depths else 0.0
        max_fill = max(fill_depths) if fill_depths else 0.0
        avg_cut = np.mean(cut_depths) if cut_depths else 0.0
        avg_fill = np.mean(fill_depths) if fill_depths else 0.0

        # Calculate balance elevation
        balance_elev = self._find_balance_elevation(polygon, cell_details)

        # Apply soil factors
        adjusted_cut = total_cut * self.swell_factor if self.swell_factor != 1.0 else None
        adjusted_fill = total_fill * self.shrink_factor if self.shrink_factor != 1.0 else None

        return EarthworkResult(
            cut_volume=total_cut,
            fill_volume=total_fill,
            net_volume=total_cut - total_fill,
            cut_area=cut_area,
            fill_area=fill_area,
            total_area=total_area,
            max_cut_depth=max_cut,
            max_fill_depth=max_fill,
            avg_cut_depth=avg_cut,
            avg_fill_depth=avg_fill,
            balance_elevation=balance_elev,
            cell_details=cell_details,
            adjusted_cut_volume=adjusted_cut,
            adjusted_fill_volume=adjusted_fill,
        )

    def calculate_sloped(
        self,
        polygon: 'Polygon',
        base_elevation: float,
        slope_percent: float,
        slope_direction: float,  # Degrees from north (0 = N, 90 = E)
    ) -> EarthworkResult:
        """
        Calculate cut/fill for a sloped grading design.

        Args:
            polygon: Shapely Polygon defining the work area
            base_elevation: Elevation at the centroid of the polygon
            slope_percent: Slope as percentage (e.g., 2.0 = 2%)
            slope_direction: Direction of downslope in degrees from north

        Returns:
            EarthworkResult with volumes and details

        Raises:
            BoundsError: If polygon doesn't intersect terrain
            EmptyResultError: If no valid cells found within polygon
            ValidationError: If slope_percent is invalid
        """
        from shapely.geometry import box
        from .validation import (
            validate_polygon_terrain_intersection,
            validate_slope_percent,
            EmptyResultError,
        )

        # Validate inputs
        validate_polygon_terrain_intersection(
            polygon, self.terrain.bounds, "terrain grid"
        )
        validate_slope_percent(slope_percent, "slope_percent")

        # Calculate slope vector
        slope_rad = np.radians(slope_direction)
        slope_ratio = slope_percent / 100.0

        # Unit vector in slope direction (pointing downhill)
        dx = np.sin(slope_rad)
        dy = np.cos(slope_rad)

        # Reference point (polygon centroid)
        centroid = polygon.centroid
        ref_x, ref_y = centroid.x, centroid.y

        def target_at(x: float, y: float) -> float:
            """Calculate target elevation at any point."""
            # Distance along slope direction from reference
            dist = (x - ref_x) * dx + (y - ref_y) * dy
            return base_elevation - dist * slope_ratio

        cell_details = []
        total_cut = 0.0
        total_fill = 0.0
        cut_area = 0.0
        fill_area = 0.0
        cut_depths = []
        fill_depths = []

        resolution = self.terrain.resolution

        min_x, min_y, max_x, max_y = polygon.bounds
        min_row, min_col = self.terrain.coord_to_cell(min_x, min_y)
        max_row, max_col = self.terrain.coord_to_cell(max_x, max_y)

        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.terrain.rows - 1, max_row)
        max_col = min(self.terrain.cols - 1, max_col)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cx, cy = self.terrain.cell_to_coord(row, col)
                cell_box = box(
                    cx - resolution / 2,
                    cy - resolution / 2,
                    cx + resolution / 2,
                    cy + resolution / 2,
                )

                if not polygon.intersects(cell_box):
                    continue

                intersection = polygon.intersection(cell_box)
                area = intersection.area

                if area < 1e-10:
                    continue

                existing = self.terrain.elevations[row, col]
                if existing == self.terrain.nodata:
                    continue

                target = target_at(cx, cy)
                delta = existing - target

                cut_depth = max(0.0, delta)
                fill_depth = max(0.0, -delta)
                cut_vol = cut_depth * area
                fill_vol = fill_depth * area

                total_cut += cut_vol
                total_fill += fill_vol

                if cut_depth > 0:
                    cut_area += area
                    cut_depths.append(cut_depth)
                if fill_depth > 0:
                    fill_area += area
                    fill_depths.append(fill_depth)

                cell_details.append(CellDetail(
                    row=row,
                    col=col,
                    x=cx,
                    y=cy,
                    existing_elevation=existing,
                    target_elevation=target,
                    cut_depth=cut_depth,
                    fill_depth=fill_depth,
                    cut_volume=cut_vol,
                    fill_volume=fill_vol,
                    area=area,
                ))

        # Check for empty results
        if not cell_details:
            raise EmptyResultError(
                "No valid terrain cells found within the polygon. "
                "This may occur if:\n"
                "  - The polygon is entirely in a nodata region\n"
                "  - The polygon is too small for the grid resolution\n"
                f"  - Grid resolution: {self.terrain.resolution}, "
                f"Polygon area: {polygon.area:.2f}"
            )

        total_area = cut_area + fill_area
        max_cut = max(cut_depths) if cut_depths else 0.0
        max_fill = max(fill_depths) if fill_depths else 0.0
        avg_cut = np.mean(cut_depths) if cut_depths else 0.0
        avg_fill = np.mean(fill_depths) if fill_depths else 0.0

        adjusted_cut = total_cut * self.swell_factor if self.swell_factor != 1.0 else None
        adjusted_fill = total_fill * self.shrink_factor if self.shrink_factor != 1.0 else None

        return EarthworkResult(
            cut_volume=total_cut,
            fill_volume=total_fill,
            net_volume=total_cut - total_fill,
            cut_area=cut_area,
            fill_area=fill_area,
            total_area=total_area,
            max_cut_depth=max_cut,
            max_fill_depth=max_fill,
            avg_cut_depth=avg_cut,
            avg_fill_depth=avg_fill,
            balance_elevation=None,  # More complex for sloped
            cell_details=cell_details,
            adjusted_cut_volume=adjusted_cut,
            adjusted_fill_volume=adjusted_fill,
        )

    def _find_balance_elevation(
        self,
        polygon: 'Polygon',
        cell_details: List[CellDetail],
    ) -> Optional[float]:
        """
        Find elevation where cut volume equals fill volume.

        Uses binary search on elevation to find balance point.
        """
        if not cell_details:
            return None

        # Get elevation range
        elevations = [c.existing_elevation for c in cell_details]
        min_elev = min(elevations)
        max_elev = max(elevations)

        def net_volume_at(target: float) -> float:
            """Calculate net volume (cut - fill) at given target elevation."""
            cut = sum(
                max(0.0, c.existing_elevation - target) * c.area
                for c in cell_details
            )
            fill = sum(
                max(0.0, target - c.existing_elevation) * c.area
                for c in cell_details
            )
            return cut - fill

        # Binary search for balance point
        low, high = min_elev, max_elev

        for _ in range(50):  # Max iterations
            mid = (low + high) / 2
            net = net_volume_at(mid)

            if abs(net) < 0.1:  # Close enough
                return mid

            if net > 0:  # More cut than fill, need higher elevation
                low = mid
            else:  # More fill than cut, need lower elevation
                high = mid

        return (low + high) / 2

    def find_optimal_elevation(
        self,
        polygon: 'Polygon',
        constraint: str = "balance",
    ) -> Tuple[float, EarthworkResult]:
        """
        Find optimal target elevation based on constraint.

        Args:
            polygon: Work area polygon
            constraint: One of:
                - "balance": Minimize |cut - fill|
                - "min_cut": Minimize total cut
                - "min_fill": Minimize total fill
                - "min_total": Minimize cut + fill

        Returns:
            Tuple of (optimal_elevation, EarthworkResult)
        """
        from scipy.optimize import minimize_scalar

        # Get elevation range from terrain within polygon
        result = self.calculate_flat(polygon, 0)  # Dummy calculation
        elevations = [c.existing_elevation for c in result.cell_details]

        if not elevations:
            raise ValueError("No terrain data within polygon")

        min_elev = min(elevations)
        max_elev = max(elevations)

        def objective(target: float) -> float:
            res = self.calculate_flat(polygon, target)
            if constraint == "balance":
                return abs(res.net_volume)
            elif constraint == "min_cut":
                return res.cut_volume
            elif constraint == "min_fill":
                return res.fill_volume
            elif constraint == "min_total":
                return res.cut_volume + res.fill_volume
            else:
                raise ValueError(f"Unknown constraint: {constraint}")

        # Optimize
        opt_result = minimize_scalar(
            objective,
            bounds=(min_elev, max_elev),
            method='bounded'
        )

        optimal_elev = opt_result.x
        final_result = self.calculate_flat(polygon, optimal_elev)

        return optimal_elev, final_result

    def generate_heatmap(
        self,
        result: EarthworkResult,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate cut/fill heatmap arrays for visualization.

        Returns:
            Tuple of (cut_grid, fill_grid, net_grid) with same shape as terrain
        """
        shape = self.terrain.shape
        cut_grid = np.zeros(shape)
        fill_grid = np.zeros(shape)

        for cell in result.cell_details:
            cut_grid[cell.row, cell.col] = cell.cut_depth
            fill_grid[cell.row, cell.col] = cell.fill_depth

        net_grid = cut_grid - fill_grid

        return cut_grid, fill_grid, net_grid
