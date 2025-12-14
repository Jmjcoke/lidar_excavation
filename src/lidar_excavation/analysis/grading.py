"""
Grading Design Module

Defines grading designs (building pads, parking lots, etc.)
for earthwork analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
import numpy as np

try:
    from shapely.geometry import Polygon, MultiPolygon, box, Point
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def require_shapely():
    """Raise ImportError if shapely is not available."""
    if not HAS_SHAPELY:
        raise ImportError("shapely is required. Install with: pip install shapely")


@dataclass
class GradingDesign(ABC):
    """
    Abstract base class for grading designs.

    A grading design defines a target surface for earthwork.
    """
    name: str
    polygon: 'Polygon'  # Boundary of the design area

    @abstractmethod
    def target_elevation_at(self, x: float, y: float) -> float:
        """Get target elevation at a specific point."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the design."""
        pass


@dataclass
class FlatGrade(GradingDesign):
    """
    Flat grading design - constant elevation across the area.

    Typical uses:
    - Building pads
    - Tennis courts
    - Level parking areas
    """
    elevation: float = 0.0

    def target_elevation_at(self, x: float, y: float) -> float:
        return self.elevation

    def description(self) -> str:
        return f"Flat grade at elevation {self.elevation:.2f}"


@dataclass
class SlopedGrade(GradingDesign):
    """
    Sloped grading design - planar surface with constant slope.

    Typical uses:
    - Parking lots (need 1-2% for drainage)
    - Driveways
    - Sports fields
    """
    base_elevation: float = 0.0  # Elevation at reference point
    slope_percent: float = 2.0   # Slope as percentage
    slope_direction: float = 0.0  # Degrees from north (0=N, 90=E, 180=S, 270=W)
    reference_point: Optional[Tuple[float, float]] = None  # Default: centroid

    def __post_init__(self):
        require_shapely()
        if self.reference_point is None:
            centroid = self.polygon.centroid
            self.reference_point = (centroid.x, centroid.y)

    def target_elevation_at(self, x: float, y: float) -> float:
        ref_x, ref_y = self.reference_point

        # Calculate direction vector (pointing downhill)
        slope_rad = np.radians(self.slope_direction)
        dx = np.sin(slope_rad)
        dy = np.cos(slope_rad)

        # Distance along slope direction from reference
        dist = (x - ref_x) * dx + (y - ref_y) * dy

        # Elevation drops as we go in the slope direction
        slope_ratio = self.slope_percent / 100.0
        return self.base_elevation - dist * slope_ratio

    def description(self) -> str:
        direction_names = {0: "N", 45: "NE", 90: "E", 135: "SE",
                         180: "S", 225: "SW", 270: "W", 315: "NW"}
        dir_name = direction_names.get(
            int(self.slope_direction) % 360,
            f"{self.slope_direction:.0f}°"
        )
        return (f"Sloped grade at {self.slope_percent:.1f}% "
                f"toward {dir_name}, base elevation {self.base_elevation:.2f}")


@dataclass
class TieredGrade(GradingDesign):
    """
    Multi-level grading design with flat terraces.

    Typical uses:
    - Hillside parking structures
    - Retaining wall systems
    - Agricultural terraces
    """
    tiers: List[Tuple['Polygon', float]] = field(default_factory=list)  # (polygon, elevation)

    def __post_init__(self):
        require_shapely()

    def add_tier(self, polygon: 'Polygon', elevation: float):
        """Add a tier to the design."""
        self.tiers.append((polygon, elevation))

    def target_elevation_at(self, x: float, y: float) -> float:
        point = Point(x, y)
        for tier_poly, elevation in self.tiers:
            if tier_poly.contains(point):
                return elevation
        # Default: return lowest tier elevation or 0
        if self.tiers:
            return min(e for _, e in self.tiers)
        return 0.0

    def description(self) -> str:
        if not self.tiers:
            return "Empty tiered design"
        elevations = [e for _, e in self.tiers]
        return f"Tiered grade with {len(self.tiers)} levels ({min(elevations):.1f} to {max(elevations):.1f})"


@dataclass
class BuildingPad:
    """
    Complete building pad design with surrounding slopes.

    Includes:
    - Flat building footprint
    - Surrounding apron with drainage slope
    - Optional retaining walls
    """
    name: str
    footprint: 'Polygon'  # Building footprint
    pad_elevation: float
    apron_width: float = 3.0  # Width of sloped area around building
    apron_slope: float = 2.0  # Percent slope away from building

    def __post_init__(self):
        require_shapely()

    @property
    def full_polygon(self) -> 'Polygon':
        """Get the full design polygon including apron."""
        return self.footprint.buffer(self.apron_width)

    def target_elevation_at(self, x: float, y: float) -> float:
        point = Point(x, y)

        # Inside building footprint: flat
        if self.footprint.contains(point):
            return self.pad_elevation

        # In apron: slope away from building
        if self.full_polygon.contains(point):
            # Distance from building edge
            dist = self.footprint.exterior.distance(point)
            slope_ratio = self.apron_slope / 100.0
            return self.pad_elevation - dist * slope_ratio

        # Outside design area
        return self.pad_elevation

    def to_grading_designs(self) -> List[GradingDesign]:
        """
        Convert to list of GradingDesign objects for analysis.

        Returns:
            - FlatGrade for the building footprint
            - SlopedGrade segments for the apron (sloped away from building)
        """
        designs: List[GradingDesign] = []

        # 1. Flat grade for building footprint
        designs.append(FlatGrade(
            name=f"{self.name}_footprint",
            polygon=self.footprint,
            elevation=self.pad_elevation,
        ))

        # 2. Sloped grade for apron area
        # The apron is the ring between footprint and full_polygon
        apron_ring = self.full_polygon.difference(self.footprint)

        if not apron_ring.is_empty and apron_ring.area > 0:
            # Get building footprint bounds to determine cardinal directions
            minx, miny, maxx, maxy = self.footprint.bounds
            apron_minx, apron_miny, apron_maxx, apron_maxy = self.full_polygon.bounds

            # Create 4 apron sectors (N, E, S, W) each sloping outward
            # North apron: slopes north (0 degrees)
            north_sector = box(minx, maxy, maxx, apron_maxy).intersection(apron_ring)
            if not north_sector.is_empty and north_sector.area > 0.01:
                designs.append(SlopedGrade(
                    name=f"{self.name}_apron_N",
                    polygon=north_sector,
                    base_elevation=self.pad_elevation,
                    slope_percent=self.apron_slope,
                    slope_direction=0.0,  # North
                    reference_point=((minx + maxx) / 2, maxy),
                ))

            # East apron: slopes east (90 degrees)
            east_sector = box(maxx, miny, apron_maxx, maxy).intersection(apron_ring)
            if not east_sector.is_empty and east_sector.area > 0.01:
                designs.append(SlopedGrade(
                    name=f"{self.name}_apron_E",
                    polygon=east_sector,
                    base_elevation=self.pad_elevation,
                    slope_percent=self.apron_slope,
                    slope_direction=90.0,  # East
                    reference_point=(maxx, (miny + maxy) / 2),
                ))

            # South apron: slopes south (180 degrees)
            south_sector = box(minx, apron_miny, maxx, miny).intersection(apron_ring)
            if not south_sector.is_empty and south_sector.area > 0.01:
                designs.append(SlopedGrade(
                    name=f"{self.name}_apron_S",
                    polygon=south_sector,
                    base_elevation=self.pad_elevation,
                    slope_percent=self.apron_slope,
                    slope_direction=180.0,  # South
                    reference_point=((minx + maxx) / 2, miny),
                ))

            # West apron: slopes west (270 degrees)
            west_sector = box(apron_minx, miny, minx, maxy).intersection(apron_ring)
            if not west_sector.is_empty and west_sector.area > 0.01:
                designs.append(SlopedGrade(
                    name=f"{self.name}_apron_W",
                    polygon=west_sector,
                    base_elevation=self.pad_elevation,
                    slope_percent=self.apron_slope,
                    slope_direction=270.0,  # West
                    reference_point=(minx, (miny + maxy) / 2),
                ))

            # Corner aprons (diagonal slopes)
            # NE corner: slopes NE (45 degrees)
            ne_corner = box(maxx, maxy, apron_maxx, apron_maxy).intersection(apron_ring)
            if not ne_corner.is_empty and ne_corner.area > 0.01:
                designs.append(SlopedGrade(
                    name=f"{self.name}_apron_NE",
                    polygon=ne_corner,
                    base_elevation=self.pad_elevation,
                    slope_percent=self.apron_slope,
                    slope_direction=45.0,
                    reference_point=(maxx, maxy),
                ))

            # SE corner
            se_corner = box(maxx, apron_miny, apron_maxx, miny).intersection(apron_ring)
            if not se_corner.is_empty and se_corner.area > 0.01:
                designs.append(SlopedGrade(
                    name=f"{self.name}_apron_SE",
                    polygon=se_corner,
                    base_elevation=self.pad_elevation,
                    slope_percent=self.apron_slope,
                    slope_direction=135.0,
                    reference_point=(maxx, miny),
                ))

            # SW corner
            sw_corner = box(apron_minx, apron_miny, minx, miny).intersection(apron_ring)
            if not sw_corner.is_empty and sw_corner.area > 0.01:
                designs.append(SlopedGrade(
                    name=f"{self.name}_apron_SW",
                    polygon=sw_corner,
                    base_elevation=self.pad_elevation,
                    slope_percent=self.apron_slope,
                    slope_direction=225.0,
                    reference_point=(minx, miny),
                ))

            # NW corner
            nw_corner = box(apron_minx, maxy, minx, apron_maxy).intersection(apron_ring)
            if not nw_corner.is_empty and nw_corner.area > 0.01:
                designs.append(SlopedGrade(
                    name=f"{self.name}_apron_NW",
                    polygon=nw_corner,
                    base_elevation=self.pad_elevation,
                    slope_percent=self.apron_slope,
                    slope_direction=315.0,
                    reference_point=(minx, maxy),
                ))

        return designs


@dataclass
class ParkingLot:
    """
    Parking lot design with drainage considerations.

    Includes:
    - Main parking surface with drainage slope
    - Optional drive aisles
    - Drainage collection points
    """
    name: str
    boundary: 'Polygon'
    base_elevation: float
    slope_percent: float = 1.5  # Typical parking lot slope
    slope_direction: float = 180.0  # Default: slopes south
    drain_locations: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        require_shapely()

    def to_grading_design(self) -> SlopedGrade:
        """Convert to SlopedGrade for analysis."""
        return SlopedGrade(
            name=self.name,
            polygon=self.boundary,
            base_elevation=self.base_elevation,
            slope_percent=self.slope_percent,
            slope_direction=self.slope_direction,
        )

    def optimal_drain_locations(
        self,
        terrain_grid=None,
        num_drains: int = 1,
        min_spacing: float = 10.0,
    ) -> List[Tuple[float, float]]:
        """
        Suggest optimal drain locations based on finished grade analysis.

        Analyzes the finished grade surface to find natural water collection
        points along the low edges of the parking lot.

        Args:
            terrain_grid: Optional terrain (for future context, not currently used)
            num_drains: Number of drain locations to suggest (default: 1)
            min_spacing: Minimum distance between drains in units (default: 10)

        Returns:
            List of (x, y) coordinates for drain placement, ordered by priority
        """
        from shapely.geometry import LineString

        centroid = self.boundary.centroid

        # Calculate slope vector (pointing downhill)
        slope_rad = np.radians(self.slope_direction)
        dx = np.sin(slope_rad)
        dy = np.cos(slope_rad)

        def finished_elevation_at(x: float, y: float) -> float:
            """Calculate finished grade elevation at a point."""
            ref_x, ref_y = centroid.x, centroid.y
            dist = (x - ref_x) * dx + (y - ref_y) * dy
            slope_ratio = self.slope_percent / 100.0
            return self.base_elevation - dist * slope_ratio

        # Sample points along the boundary to find lowest points
        boundary_coords = list(self.boundary.exterior.coords)
        boundary_elevations = []

        for coord in boundary_coords[:-1]:  # Exclude duplicate closing point
            x, y = coord
            elev = finished_elevation_at(x, y)
            boundary_elevations.append((x, y, elev))

        # Sort by elevation (lowest first = best drain locations)
        boundary_elevations.sort(key=lambda p: p[2])

        # Select drain locations with minimum spacing
        drains: List[Tuple[float, float]] = []

        for x, y, elev in boundary_elevations:
            if len(drains) >= num_drains:
                break

            # Check spacing from existing drains
            too_close = False
            for dx_prev, dy_prev in drains:
                dist = np.sqrt((x - dx_prev)**2 + (y - dy_prev)**2)
                if dist < min_spacing:
                    too_close = True
                    break

            if not too_close:
                drains.append((x, y))

        # Fallback: if we couldn't find enough spaced drains, add lowest points
        if len(drains) < num_drains:
            for x, y, elev in boundary_elevations:
                if (x, y) not in drains:
                    drains.append((x, y))
                    if len(drains) >= num_drains:
                        break

        return drains

    def analyze_drainage(self) -> dict:
        """
        Analyze drainage characteristics of the parking lot.

        Returns:
            Dictionary with drainage analysis:
            - slope_percent: Design slope
            - slope_direction: Direction water flows
            - low_point: Coordinates of lowest finished grade point
            - high_point: Coordinates of highest finished grade point
            - elevation_drop: Total elevation change across lot
            - suggested_drains: List of suggested drain locations
            - catchment_area: Area draining to each suggested drain
        """
        centroid = self.boundary.centroid
        slope_rad = np.radians(self.slope_direction)
        dx = np.sin(slope_rad)
        dy = np.cos(slope_rad)

        def finished_elevation_at(x: float, y: float) -> float:
            dist = (x - centroid.x) * dx + (y - centroid.y) * dy
            return self.base_elevation - dist * (self.slope_percent / 100.0)

        # Find high and low points by sampling boundary
        boundary_coords = list(self.boundary.exterior.coords)
        sample_elevations = [
            (x, y, finished_elevation_at(x, y))
            for x, y in boundary_coords[:-1]
        ]

        if not sample_elevations:
            return {"error": "No points on boundary"}

        sample_elevations.sort(key=lambda p: p[2])
        low_point = sample_elevations[0]
        high_point = sample_elevations[-1]

        return {
            "slope_percent": self.slope_percent,
            "slope_direction": self.slope_direction,
            "low_point": (low_point[0], low_point[1]),
            "low_elevation": low_point[2],
            "high_point": (high_point[0], high_point[1]),
            "high_elevation": high_point[2],
            "elevation_drop": high_point[2] - low_point[2],
            "suggested_drains": self.optimal_drain_locations(num_drains=2),
            "catchment_area": self.boundary.area,
        }


class SiteDesign:
    """
    Complete site design combining multiple grading elements.
    """

    def __init__(self, name: str):
        require_shapely()
        self.name = name
        self.elements: List[Union[GradingDesign, BuildingPad, ParkingLot]] = []

    def add_element(self, element: Union[GradingDesign, BuildingPad, ParkingLot]):
        """Add a design element to the site."""
        self.elements.append(element)

    def get_all_polygons(self) -> List['Polygon']:
        """Get all design polygons."""
        polygons = []
        for elem in self.elements:
            if isinstance(elem, BuildingPad):
                polygons.append(elem.full_polygon)
            elif isinstance(elem, ParkingLot):
                polygons.append(elem.boundary)
            else:
                polygons.append(elem.polygon)
        return polygons

    def combined_boundary(self) -> 'Polygon':
        """Get the combined boundary of all elements."""
        polygons = self.get_all_polygons()
        if not polygons:
            raise ValueError("No elements in site design")
        return unary_union(polygons)

    def total_area(self) -> float:
        """Total area of all design elements."""
        return sum(p.area for p in self.get_all_polygons())


def create_rectangle(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation: float = 0.0,
) -> 'Polygon':
    """
    Create a rectangular polygon.

    Args:
        center_x, center_y: Center coordinates
        width, height: Rectangle dimensions
        rotation: Rotation in degrees (counterclockwise)

    Returns:
        Shapely Polygon
    """
    require_shapely()
    from shapely import affinity

    # Create axis-aligned rectangle
    rect = box(
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )

    # Rotate if needed
    if rotation != 0:
        rect = affinity.rotate(rect, rotation, origin='centroid')

    return rect


def create_polygon_from_coords(coords: List[Tuple[float, float]]) -> 'Polygon':
    """
    Create a polygon from a list of (x, y) coordinates.

    Args:
        coords: List of (x, y) tuples defining polygon vertices

    Returns:
        Shapely Polygon
    """
    require_shapely()
    return Polygon(coords)
