"""
Basic Excavation Analysis Example

This example demonstrates:
1. Loading or generating point cloud data
2. Creating a terrain grid (DEM)
3. Defining a building pad and parking lot
4. Calculating cut/fill volumes
5. Visualizing results

Run from the project root:
    python examples/basic_analysis.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from shapely.geometry import Polygon, box

from lidar_excavation.io.point_cloud import generate_sample_terrain, PointCloudLoader
from lidar_excavation.core.terrain import TerrainGrid, InterpolationMethod
from lidar_excavation.core.volume import VolumeCalculator
from lidar_excavation.analysis.grading import (
    FlatGrade,
    SlopedGrade,
    BuildingPad,
    ParkingLot,
    create_rectangle,
)


def main():
    print("=" * 60)
    print("LIDAR EXCAVATION ANALYSIS - EXAMPLE")
    print("=" * 60)

    # =========================================================================
    # Step 1: Generate or load point cloud
    # =========================================================================
    print("\n[1] Generating sample terrain...")

    # Generate synthetic terrain (replace with PointCloudLoader.load() for real data)
    point_cloud = generate_sample_terrain(
        size=(200.0, 150.0),     # 200m x 150m site
        resolution=1.0,          # 1m point spacing
        base_elevation=100.0,    # Base elevation ~100m
        noise_scale=1.5,         # Some surface variation
        hill_height=8.0,         # Hills up to 8m high
        seed=42,                 # Reproducible results
    )

    bounds_min, bounds_max = point_cloud.bounds
    print(f"   Points: {point_cloud.num_points:,}")
    print(f"   X range: {bounds_min[0]:.1f} to {bounds_max[0]:.1f}")
    print(f"   Y range: {bounds_min[1]:.1f} to {bounds_max[1]:.1f}")
    print(f"   Z range: {bounds_min[2]:.1f} to {bounds_max[2]:.1f}")

    # =========================================================================
    # Step 2: Create terrain grid (DEM)
    # =========================================================================
    print("\n[2] Creating terrain grid...")

    terrain = TerrainGrid.from_point_cloud(
        point_cloud,
        resolution=1.0,                      # 1m grid cells
        method=InterpolationMethod.MEAN,     # Average elevation in each cell
        ground_only=True,                    # Only use ground points
    )

    print(f"   Grid size: {terrain.rows} x {terrain.cols}")
    stats = terrain.statistics()
    print(f"   Elevation range: {stats['min_elevation']:.1f} to {stats['max_elevation']:.1f}")
    print(f"   Mean slope: {stats['mean_slope']:.1f} degrees")

    # =========================================================================
    # Step 3: Define design areas
    # =========================================================================
    print("\n[3] Defining design areas...")

    # Building pad: 30m x 20m rectangle at center of site
    building_footprint = create_rectangle(
        center_x=100.0,
        center_y=75.0,
        width=30.0,
        height=20.0,
    )

    # Parking lot: 40m x 25m rectangle adjacent to building
    parking_boundary = create_rectangle(
        center_x=100.0,
        center_y=35.0,  # South of building
        width=40.0,
        height=25.0,
    )

    print(f"   Building footprint: {building_footprint.area:.0f} sq meters")
    print(f"   Parking lot: {parking_boundary.area:.0f} sq meters")

    # =========================================================================
    # Step 4: Calculate earthwork for building pad (flat grade)
    # =========================================================================
    print("\n[4] Analyzing building pad (flat grade)...")

    calculator = VolumeCalculator(
        terrain,
        swell_factor=1.2,    # Excavated soil expands 20%
        shrink_factor=0.9,   # Fill compacts to 90%
    )

    # First, find optimal elevation for balanced earthwork
    optimal_elev, _ = calculator.find_optimal_elevation(
        building_footprint,
        constraint="balance"
    )
    print(f"   Optimal elevation for balance: {optimal_elev:.2f}m")

    # Calculate at optimal elevation
    building_result = calculator.calculate_flat(
        building_footprint,
        target_elevation=optimal_elev,
    )

    print(f"\n   BUILDING PAD RESULTS:")
    print(f"   Target elevation: {optimal_elev:.2f}m")
    print(f"   Cut volume:  {building_result.cut_volume:,.0f} cubic meters")
    print(f"   Fill volume: {building_result.fill_volume:,.0f} cubic meters")
    print(f"   Net volume:  {building_result.net_volume:,.0f} cubic meters")
    print(f"   Max cut depth:  {building_result.max_cut_depth:.2f}m")
    print(f"   Max fill depth: {building_result.max_fill_depth:.2f}m")

    if building_result.adjusted_cut_volume:
        print(f"\n   With soil factors:")
        print(f"   Adjusted cut (swell):   {building_result.adjusted_cut_volume:,.0f} cubic meters")
        print(f"   Adjusted fill (shrink): {building_result.adjusted_fill_volume:,.0f} cubic meters")

    # =========================================================================
    # Step 5: Calculate earthwork for parking lot (sloped grade)
    # =========================================================================
    print("\n[5] Analyzing parking lot (sloped grade for drainage)...")

    # Parking lot needs slope for drainage (typically 1-2%)
    parking_result = calculator.calculate_sloped(
        parking_boundary,
        base_elevation=optimal_elev - 0.5,  # Slightly lower than building
        slope_percent=1.5,                   # 1.5% slope
        slope_direction=180.0,               # Slopes south
    )

    print(f"\n   PARKING LOT RESULTS:")
    print(f"   Slope: 1.5% toward south")
    print(f"   Cut volume:  {parking_result.cut_volume:,.0f} cubic meters")
    print(f"   Fill volume: {parking_result.fill_volume:,.0f} cubic meters")
    print(f"   Net volume:  {parking_result.net_volume:,.0f} cubic meters")

    # =========================================================================
    # Step 6: Combined site summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SITE SUMMARY")
    print("=" * 60)

    total_cut = building_result.cut_volume + parking_result.cut_volume
    total_fill = building_result.fill_volume + parking_result.fill_volume
    total_net = total_cut - total_fill

    print(f"Total Cut:     {total_cut:,.0f} cubic meters")
    print(f"Total Fill:    {total_fill:,.0f} cubic meters")
    print(f"Net Movement:  {total_net:,.0f} cubic meters")

    if total_net > 0:
        print(f"\n=> {total_net:,.0f} cubic meters of material must be EXPORTED from site")
    else:
        print(f"\n=> {abs(total_net):,.0f} cubic meters of material must be IMPORTED to site")

    # =========================================================================
    # Step 7: Visualization (if matplotlib available)
    # =========================================================================
    print("\n[6] Generating visualization...")

    try:
        from lidar_excavation.utils.visualization import (
            plot_terrain,
            plot_earthwork,
            plot_cross_section,
            create_report_figure,
        )
        import matplotlib.pyplot as plt

        # Create comprehensive report figure
        # Combine building and parking polygons for display
        from shapely.ops import unary_union
        combined_polygon = unary_union([building_footprint, parking_boundary])

        fig = create_report_figure(
            terrain,
            building_result,
            polygon=combined_polygon,
            cross_section_line=((50, 75), (150, 75)),  # E-W through building
            target_elevation=optimal_elev,
        )

        # Save figure
        output_path = Path(__file__).parent / "analysis_report.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Report saved to: {output_path}")

        # Show interactively
        plt.show()

    except ImportError:
        print("   (matplotlib not available - skipping visualization)")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


def demo_with_real_file():
    """
    Example of analyzing a real LAS/LAZ file.

    Uncomment and modify paths to use with your data.
    """
    # Load real LIDAR data
    # pc = PointCloudLoader.load("path/to/your/terrain.las")
    #
    # # Filter to ground points only
    # ground_points = pc.filter_by_classification([2])  # Class 2 = Ground
    #
    # # Remove outliers
    # clean_points = ground_points.remove_statistical_outliers(k_neighbors=20, std_ratio=2.0)
    #
    # # Create terrain grid
    # terrain = TerrainGrid.from_point_cloud(clean_points, resolution=0.5)
    #
    # # Define your work area (adjust coordinates to match your data)
    # work_area = Polygon([
    #     (1000, 2000),
    #     (1100, 2000),
    #     (1100, 2050),
    #     (1000, 2050),
    # ])
    #
    # # Calculate volumes
    # calculator = VolumeCalculator(terrain)
    # result = calculator.calculate_flat(work_area, target_elevation=150.0)
    # print(result.summary())
    pass


if __name__ == "__main__":
    main()
