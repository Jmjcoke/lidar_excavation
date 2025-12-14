"""
Command Line Interface for LIDAR Excavation Analysis

Usage:
    lidar-excavation analyze <input> --polygon <coords> --elevation <target>
    lidar-excavation info <input>
    lidar-excavation generate-sample --output <file>
"""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np

from .io.point_cloud import PointCloudLoader, PointCloud, generate_sample_terrain
from .core.terrain import TerrainGrid, InterpolationMethod
from .core.volume import VolumeCalculator, EarthworkResult


@click.group()
@click.version_option(version="0.1.0")
def main():
    """LIDAR Excavation Analysis Tool

    Analyze LIDAR point clouds to calculate cut/fill volumes
    for construction excavation planning.
    """
    pass


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
def info(input_file: str):
    """Display information about a point cloud file."""
    click.echo(f"Loading: {input_file}")

    try:
        pc = PointCloudLoader.load(input_file)
    except Exception as e:
        click.echo(f"Error loading file: {e}", err=True)
        sys.exit(1)

    bounds_min, bounds_max = pc.bounds

    click.echo("\n" + "=" * 50)
    click.echo("POINT CLOUD INFO")
    click.echo("=" * 50)
    click.echo(f"File:           {input_file}")
    click.echo(f"Points:         {pc.num_points:,}")
    click.echo(f"")
    click.echo(f"Bounds:")
    click.echo(f"  X:            {bounds_min[0]:.2f} to {bounds_max[0]:.2f}")
    click.echo(f"  Y:            {bounds_min[1]:.2f} to {bounds_max[1]:.2f}")
    click.echo(f"  Z:            {bounds_min[2]:.2f} to {bounds_max[2]:.2f}")
    click.echo(f"")
    click.echo(f"Extent:")
    click.echo(f"  Width:        {bounds_max[0] - bounds_min[0]:.2f}")
    click.echo(f"  Height:       {bounds_max[1] - bounds_min[1]:.2f}")
    click.echo(f"  Z Range:      {bounds_max[2] - bounds_min[2]:.2f}")

    if pc.classification is not None:
        unique, counts = np.unique(pc.classification, return_counts=True)
        click.echo(f"\nClassifications:")
        class_names = {
            1: "Unclassified",
            2: "Ground",
            3: "Low Vegetation",
            4: "Medium Vegetation",
            5: "High Vegetation",
            6: "Building",
            7: "Noise",
            9: "Water",
        }
        for cls, count in zip(unique, counts):
            name = class_names.get(cls, f"Class {cls}")
            pct = count / pc.num_points * 100
            click.echo(f"  {name}: {count:,} ({pct:.1f}%)")

    click.echo("=" * 50)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--resolution', '-r', default=1.0, help='Grid cell size (default: 1.0)')
@click.option('--elevation', '-e', required=True, type=float, help='Target elevation')
@click.option('--polygon', '-p', type=str,
              help='Polygon coordinates as JSON: [[x1,y1],[x2,y2],...]')
@click.option('--rect', type=str,
              help='Rectangle as "x,y,width,height" (alternative to polygon)')
@click.option('--output', '-o', type=click.Path(), help='Output JSON file for results')
@click.option('--export-csv', type=click.Path(), help='Export cell details to CSV file')
@click.option('--export-geojson', type=click.Path(), help='Export cut/fill zones to GeoJSON')
@click.option('--plot', is_flag=True, help='Show visualization plots')
@click.option('--report', type=click.Path(), help='Save PDF/PNG report to file')
@click.option('--ground-only/--all-points', default=True,
              help='Use only ground-classified points (default: ground only)')
@click.option('--swell-factor', default=1.0, help='Soil swell factor for cut volumes')
@click.option('--shrink-factor', default=1.0, help='Soil shrink factor for fill volumes')
def analyze(
    input_file: str,
    resolution: float,
    elevation: float,
    polygon: Optional[str],
    rect: Optional[str],
    output: Optional[str],
    export_csv: Optional[str],
    export_geojson: Optional[str],
    plot: bool,
    report: Optional[str],
    ground_only: bool,
    swell_factor: float,
    shrink_factor: float,
):
    """Analyze point cloud and calculate cut/fill volumes.

    Requires either --polygon or --rect to define the work area.

    Examples:

        # Analyze with rectangle
        lidar-excavation analyze terrain.las -e 100.0 --rect "50,50,30,20"

        # Analyze with polygon
        lidar-excavation analyze terrain.las -e 100.0 \\
            --polygon "[[0,0],[100,0],[100,50],[0,50]]"
    """
    from shapely.geometry import Polygon as ShapelyPolygon, box

    # Load point cloud
    click.echo(f"Loading point cloud: {input_file}")
    try:
        pc = PointCloudLoader.load(input_file)
        click.echo(f"  Loaded {pc.num_points:,} points")
    except Exception as e:
        click.echo(f"Error loading file: {e}", err=True)
        sys.exit(1)

    # Parse work area polygon
    work_polygon = None

    if polygon:
        try:
            coords = json.loads(polygon)
            work_polygon = ShapelyPolygon(coords)
            click.echo(f"  Work area: Custom polygon ({work_polygon.area:.1f} sq units)")
        except Exception as e:
            click.echo(f"Error parsing polygon: {e}", err=True)
            sys.exit(1)

    elif rect:
        try:
            parts = rect.split(',')
            if len(parts) != 4:
                raise ValueError(
                    f"Rectangle must have exactly 4 values (got {len(parts)}). "
                    "Format: x,y,width,height"
                )

            try:
                x, y, w, h = [float(p.strip()) for p in parts]
            except ValueError as e:
                raise ValueError(
                    f"Rectangle values must be numbers. "
                    f"Got: {parts}. Error: {e}"
                )

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Rectangle width and height must be positive. "
                    f"Got width={w}, height={h}"
                )

            work_polygon = box(x, y, x + w, y + h)
            click.echo(f"  Work area: Rectangle {w}x{h} at ({x},{y})")
        except ValueError as e:
            click.echo(f"Error parsing rectangle: {e}", err=True)
            click.echo('Example: --rect "50,50,100,75" for a 100x75 rectangle at position (50,50)', err=True)
            sys.exit(1)

    else:
        click.echo("Error: Must specify either --polygon or --rect", err=True)
        sys.exit(1)

    # Validate resolution
    if resolution <= 0:
        click.echo(
            f"Error: Resolution must be positive (got {resolution}). "
            "Typical values are 0.5-5.0 meters.",
            err=True
        )
        sys.exit(1)

    # Generate terrain grid
    click.echo(f"Generating terrain grid (resolution: {resolution})...")
    try:
        terrain = TerrainGrid.from_point_cloud(
            pc,
            resolution=resolution,
            method=InterpolationMethod.MEAN,
            ground_only=ground_only,
        )
        click.echo(f"  Grid size: {terrain.rows} x {terrain.cols}")
    except Exception as e:
        click.echo(f"Error generating terrain: {e}", err=True)
        sys.exit(1)

    # Calculate volumes
    click.echo(f"Calculating earthwork (target elevation: {elevation})...")
    calculator = VolumeCalculator(
        terrain,
        swell_factor=swell_factor,
        shrink_factor=shrink_factor,
    )

    try:
        result = calculator.calculate_flat(work_polygon, elevation)
    except Exception as e:
        click.echo(f"Error calculating volumes: {e}", err=True)
        sys.exit(1)

    # Display results
    click.echo("\n" + result.summary())

    # Save JSON output
    if output:
        try:
            from .core.validation import validate_output_path
            output_path = validate_output_path(output, "output JSON file")
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            click.echo(f"\nResults saved to: {output}")
        except Exception as e:
            click.echo(f"Error saving output: {e}", err=True)
            sys.exit(1)

    # Export cell details to CSV
    if export_csv:
        try:
            from .io.exporters import export_cell_details_csv
            from .core.validation import validate_output_path
            csv_path = validate_output_path(export_csv, "CSV output")
            export_cell_details_csv(result, csv_path)
            click.echo(f"Cell details exported to: {export_csv}")
        except Exception as e:
            click.echo(f"Error exporting CSV: {e}", err=True)
            sys.exit(1)

    # Export cut/fill zones to GeoJSON
    if export_geojson:
        try:
            from .io.exporters import export_cut_fill_zones_geojson
            from .core.validation import validate_output_path
            geojson_path = validate_output_path(export_geojson, "GeoJSON output")
            export_cut_fill_zones_geojson(result, terrain, geojson_path)
            click.echo(f"Cut/fill zones exported to: {export_geojson}")
        except Exception as e:
            click.echo(f"Error exporting GeoJSON: {e}", err=True)
            sys.exit(1)

    # Show plots
    if plot or report:
        try:
            from .utils.visualization import create_report_figure, save_report
            import matplotlib.pyplot as plt

            fig = create_report_figure(
                terrain,
                result,
                polygon=work_polygon,
                target_elevation=elevation,
            )

            if report:
                save_report(fig, report)

            if plot:
                plt.show()

        except ImportError:
            click.echo("Warning: matplotlib required for plotting", err=True)


@main.command()
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output file path (.las or .xyz)')
@click.option('--size', default="100,100", help='Terrain size as "width,height" (default: 100,100)')
@click.option('--resolution', '-r', default=1.0, help='Point spacing (default: 1.0)')
@click.option('--base-elevation', default=100.0, help='Base elevation (default: 100)')
@click.option('--hill-height', default=10.0, help='Maximum hill height (default: 10)')
@click.option('--seed', default=42, help='Random seed (default: 42)')
def generate_sample(
    output: str,
    size: str,
    resolution: float,
    base_elevation: float,
    hill_height: float,
    seed: int,
):
    """Generate a sample terrain point cloud for testing.

    Creates synthetic terrain with gentle hills and random variation.

    Example:
        lidar-excavation generate-sample -o sample.xyz --size 200,200
    """
    # Parse size
    try:
        width, height = [float(x) for x in size.split(',')]
    except ValueError:
        click.echo("Error: Size must be 'width,height'", err=True)
        sys.exit(1)

    click.echo(f"Generating sample terrain...")
    click.echo(f"  Size: {width} x {height}")
    click.echo(f"  Resolution: {resolution}")
    click.echo(f"  Base elevation: {base_elevation}")

    pc = generate_sample_terrain(
        size=(width, height),
        resolution=resolution,
        base_elevation=base_elevation,
        noise_scale=2.0,
        hill_height=hill_height,
        seed=seed,
    )

    click.echo(f"  Generated {pc.num_points:,} points")

    # Save to file
    output_path = Path(output)

    if output_path.suffix.lower() in ['.las', '.laz']:
        try:
            import laspy
            header = laspy.LasHeader(point_format=0, version="1.2")
            las = laspy.LasData(header)
            las.x = pc.x
            las.y = pc.y
            las.z = pc.z
            las.classification = pc.classification
            las.write(output)
            click.echo(f"Saved to: {output}")
        except ImportError:
            click.echo("Error: laspy required for LAS output. Use .xyz instead.", err=True)
            sys.exit(1)
    else:
        # Save as XYZ text file
        with open(output, 'w') as f:
            for i in range(pc.num_points):
                f.write(f"{pc.x[i]:.3f} {pc.y[i]:.3f} {pc.z[i]:.3f}")
                if pc.classification is not None:
                    f.write(f" {pc.classification[i]}")
                f.write("\n")
        click.echo(f"Saved to: {output}")


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--resolution', '-r', default=1.0, help='Grid cell size')
@click.option('--polygon', '-p', type=str, required=True,
              help='Polygon coordinates as JSON')
@click.option('--constraint', '-c', type=click.Choice(['balance', 'min_cut', 'min_fill', 'min_total']),
              default='balance', help='Optimization constraint')
def optimize(
    input_file: str,
    resolution: float,
    polygon: str,
    constraint: str,
):
    """Find optimal grading elevation.

    Finds the elevation that minimizes earthwork based on constraint:

    \b
    - balance:   Minimize |cut - fill| (balanced earthwork)
    - min_cut:   Minimize excavation
    - min_fill:  Minimize fill material needed
    - min_total: Minimize total earthwork (cut + fill)

    Example:
        lidar-excavation optimize terrain.las --polygon "[[0,0],[50,0],[50,30],[0,30]]" -c balance
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    # Load and process
    click.echo(f"Loading: {input_file}")
    pc = PointCloudLoader.load(input_file)

    coords = json.loads(polygon)
    work_polygon = ShapelyPolygon(coords)

    click.echo(f"Generating terrain grid...")
    terrain = TerrainGrid.from_point_cloud(pc, resolution=resolution)

    click.echo(f"Finding optimal elevation (constraint: {constraint})...")
    calculator = VolumeCalculator(terrain)

    optimal_elev, result = calculator.find_optimal_elevation(work_polygon, constraint)

    click.echo("\n" + "=" * 50)
    click.echo(f"OPTIMIZATION RESULT")
    click.echo("=" * 50)
    click.echo(f"Constraint:         {constraint}")
    click.echo(f"Optimal Elevation:  {optimal_elev:.2f}")
    click.echo(f"")
    click.echo(f"At this elevation:")
    click.echo(f"  Cut Volume:       {result.cut_volume:,.1f}")
    click.echo(f"  Fill Volume:      {result.fill_volume:,.1f}")
    click.echo(f"  Net Volume:       {result.net_volume:,.1f}")
    click.echo("=" * 50)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output GeoTIFF file')
@click.option('--resolution', '-r', default=1.0, help='Grid cell size')
def to_dem(
    input_file: str,
    output: str,
    resolution: float,
):
    """Convert point cloud to DEM (GeoTIFF).

    Example:
        lidar-excavation to-dem terrain.las -o terrain.tif -r 0.5
    """
    click.echo(f"Loading: {input_file}")
    pc = PointCloudLoader.load(input_file)

    click.echo(f"Generating terrain grid (resolution: {resolution})...")
    terrain = TerrainGrid.from_point_cloud(pc, resolution=resolution)

    click.echo(f"Exporting to GeoTIFF...")
    try:
        terrain.to_geotiff(output)
        click.echo(f"Saved to: {output}")
    except ImportError:
        click.echo("Error: rasterio required for GeoTIFF export", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
