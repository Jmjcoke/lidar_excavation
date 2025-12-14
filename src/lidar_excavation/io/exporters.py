"""
Export utilities for earthwork analysis results.

Provides CSV, GeoJSON, and optional raster exports.
Uses only standard library for CSV/JSON to avoid dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any, List

if TYPE_CHECKING:
    from ..core.volume import EarthworkResult
    from ..core.terrain import TerrainGrid
    from shapely.geometry import Polygon


def export_cell_details_csv(
    result: 'EarthworkResult',
    filepath: str,
    include_header: bool = True,
) -> None:
    """
    Export cell-level cut/fill details to CSV.

    Columns: row, col, x, y, existing_elevation, target_elevation,
             cut_depth, fill_depth, cut_volume, fill_volume, area

    Args:
        result: EarthworkResult from volume calculation
        filepath: Output CSV file path
        include_header: Whether to include column header row (default: True)
    """
    filepath = Path(filepath)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if include_header:
            writer.writerow([
                'row', 'col', 'x', 'y',
                'existing_elevation', 'target_elevation',
                'cut_depth', 'fill_depth',
                'cut_volume', 'fill_volume', 'area'
            ])

        for cell in result.cell_details:
            writer.writerow([
                cell.row, cell.col,
                f"{cell.x:.6f}", f"{cell.y:.6f}",
                f"{cell.existing_elevation:.4f}",
                f"{cell.target_elevation:.4f}",
                f"{cell.cut_depth:.4f}",
                f"{cell.fill_depth:.4f}",
                f"{cell.cut_volume:.4f}",
                f"{cell.fill_volume:.4f}",
                f"{cell.area:.6f}",
            ])


def export_summary_json(
    result: 'EarthworkResult',
    filepath: str,
    include_cell_details: bool = False,
    indent: int = 2,
) -> None:
    """
    Export earthwork summary to JSON.

    Args:
        result: EarthworkResult from volume calculation
        filepath: Output JSON file path
        include_cell_details: Include full cell details (can be large)
        indent: JSON indentation level (default: 2)
    """
    data = result.to_dict()

    if include_cell_details:
        data['cell_details'] = [
            {
                'row': c.row,
                'col': c.col,
                'x': c.x,
                'y': c.y,
                'existing_elevation': c.existing_elevation,
                'target_elevation': c.target_elevation,
                'cut_depth': c.cut_depth,
                'fill_depth': c.fill_depth,
                'cut_volume': c.cut_volume,
                'fill_volume': c.fill_volume,
                'area': c.area,
            }
            for c in result.cell_details
        ]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def export_boundary_geojson(
    polygon: 'Polygon',
    filepath: str,
    properties: Optional[Dict[str, Any]] = None,
    crs: Optional[str] = None,
) -> None:
    """
    Export design boundary polygon to GeoJSON.

    Args:
        polygon: Shapely Polygon defining the work area
        filepath: Output GeoJSON file path
        properties: Optional properties dict to attach to feature
        crs: Optional CRS string (added as foreign member)
    """
    # Extract coordinates from Shapely polygon
    exterior_coords = list(polygon.exterior.coords)

    # Build GeoJSON structure
    geojson: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [exterior_coords]
            },
            "properties": properties or {}
        }]
    }

    # Add CRS as foreign member (GeoJSON 2008 spec allows this)
    if crs:
        geojson["crs"] = {
            "type": "name",
            "properties": {"name": crs}
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)


def export_cut_fill_zones_geojson(
    result: 'EarthworkResult',
    terrain: 'TerrainGrid',
    filepath: str,
    threshold: float = 0.01,
) -> None:
    """
    Export cut and fill zones as GeoJSON polygons.

    Creates separate features for cut and fill areas with summary properties.

    Args:
        result: EarthworkResult from volume calculation
        terrain: TerrainGrid used in the calculation
        filepath: Output GeoJSON file path
        threshold: Minimum depth to include (default: 0.01)
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    cut_cells = []
    fill_cells = []

    resolution = terrain.resolution

    for cell in result.cell_details:
        cx, cy = terrain.cell_to_coord(cell.row, cell.col)
        cell_box = box(
            cx - resolution / 2,
            cy - resolution / 2,
            cx + resolution / 2,
            cy + resolution / 2,
        )

        if cell.cut_depth > threshold:
            cut_cells.append(cell_box)
        elif cell.fill_depth > threshold:
            fill_cells.append(cell_box)

    features: List[Dict[str, Any]] = []

    if cut_cells:
        cut_polygon = unary_union(cut_cells)
        features.append({
            "type": "Feature",
            "geometry": cut_polygon.__geo_interface__,
            "properties": {
                "type": "cut",
                "volume": result.cut_volume,
                "area": result.cut_area,
                "max_depth": result.max_cut_depth,
                "avg_depth": result.avg_cut_depth,
            }
        })

    if fill_cells:
        fill_polygon = unary_union(fill_cells)
        features.append({
            "type": "Feature",
            "geometry": fill_polygon.__geo_interface__,
            "properties": {
                "type": "fill",
                "volume": result.fill_volume,
                "area": result.fill_area,
                "max_depth": result.max_fill_depth,
                "avg_depth": result.avg_fill_depth,
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "total_area": result.total_area,
            "net_volume": result.net_volume,
            "cut_volume": result.cut_volume,
            "fill_volume": result.fill_volume,
        }
    }

    # Add CRS if available
    if terrain.crs:
        geojson["crs"] = {
            "type": "name",
            "properties": {"name": terrain.crs}
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)


def export_cut_fill_raster(
    result: 'EarthworkResult',
    terrain: 'TerrainGrid',
    filepath: str,
    raster_type: str = "net",
) -> None:
    """
    Export cut/fill depths as GeoTIFF raster.

    Requires rasterio (optional dependency).

    Args:
        result: EarthworkResult from volume calculation
        terrain: TerrainGrid used in the calculation
        filepath: Output GeoTIFF file path
        raster_type: One of "cut", "fill", or "net" (default: "net")
            - "cut": Cut depths only (positive values)
            - "fill": Fill depths only (positive values)
            - "net": Cut minus fill (positive = cut, negative = fill)
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        raise ImportError(
            "rasterio required for raster export. "
            "Install with: pip install rasterio"
        )

    import numpy as np

    # Create raster grid initialized with nodata
    shape = terrain.shape
    data = np.full(shape, terrain.nodata, dtype=np.float64)

    for cell in result.cell_details:
        if raster_type == "cut":
            data[cell.row, cell.col] = cell.cut_depth
        elif raster_type == "fill":
            data[cell.row, cell.col] = cell.fill_depth
        else:  # net
            data[cell.row, cell.col] = cell.cut_depth - cell.fill_depth

    transform = from_origin(
        terrain.origin[0],
        terrain.origin[1] + terrain.rows * terrain.resolution,
        terrain.resolution,
        terrain.resolution
    )

    # Flip vertically for GeoTIFF convention
    data = np.flipud(data)

    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=terrain.rows,
        width=terrain.cols,
        count=1,
        dtype=data.dtype,
        crs=terrain.crs,
        transform=transform,
        nodata=terrain.nodata,
    ) as dst:
        dst.write(data, 1)
