"""
LIDAR Excavation Analysis - Web Interface

A Streamlit app for analyzing LIDAR point clouds and calculating
cut/fill volumes for construction excavation planning.
"""

import json
import tempfile
from pathlib import Path
from io import BytesIO

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="LIDAR Excavation Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def load_point_cloud(uploaded_file):
    """Load point cloud from uploaded file."""
    from .io.point_cloud import PointCloudLoader

    # Save to temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        pc = PointCloudLoader.load(tmp_path)
        return pc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def create_terrain_grid(pc, resolution, ground_only):
    """Create terrain grid from point cloud."""
    from .core.terrain import TerrainGrid, InterpolationMethod

    return TerrainGrid.from_point_cloud(
        pc,
        resolution=resolution,
        method=InterpolationMethod.MEAN,
        ground_only=ground_only,
    )


def create_3d_terrain_plot(terrain, result=None, title="Terrain Surface"):
    """Create interactive 3D terrain visualization."""
    # Sample the terrain for performance (max 200x200)
    step = max(1, max(terrain.rows, terrain.cols) // 200)
    z_data = terrain.elevation[::step, ::step]

    # Create coordinate grids
    x = np.linspace(terrain.bounds[0], terrain.bounds[2], z_data.shape[1])
    y = np.linspace(terrain.bounds[1], terrain.bounds[3], z_data.shape[0])

    fig = go.Figure()

    # Add terrain surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z_data,
        colorscale='earth',
        name='Terrain',
        showscale=True,
        colorbar=dict(title='Elevation', x=1.02),
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Elevation',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )

    return fig


def create_2d_terrain_plot(terrain, title="Terrain Elevation"):
    """Create 2D heatmap of terrain."""
    fig = px.imshow(
        terrain.elevation,
        origin='lower',
        color_continuous_scale='earth',
        labels={'color': 'Elevation'},
        title=title,
    )
    fig.update_layout(height=500)
    return fig


def create_cut_fill_plot(result, terrain):
    """Create cut/fill visualization."""
    # Calculate cut/fill grid
    cut_fill = np.zeros((terrain.rows, terrain.cols))

    if hasattr(result, 'cell_details') and result.cell_details:
        for cell in result.cell_details:
            r, c = cell.get('row', 0), cell.get('col', 0)
            if 0 <= r < terrain.rows and 0 <= c < terrain.cols:
                if cell.get('cut_volume', 0) > 0:
                    cut_fill[r, c] = cell['cut_depth']
                elif cell.get('fill_volume', 0) > 0:
                    cut_fill[r, c] = -cell['fill_depth']
    else:
        # Fallback: calculate from target elevation
        target = result.target_elevation if hasattr(result, 'target_elevation') else None
        if target:
            diff = terrain.elevation - target
            cut_fill = np.where(np.isnan(diff), 0, diff)

    # Create diverging colorscale (red=cut, blue=fill)
    fig = px.imshow(
        cut_fill,
        origin='lower',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        labels={'color': 'Cut (+) / Fill (-)'},
        title='Cut/Fill Analysis',
    )
    fig.update_layout(height=500)
    return fig


def create_volume_chart(result):
    """Create bar chart of volumes."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Cut', 'Fill', 'Net'],
        y=[result.cut_volume, result.fill_volume, abs(result.net_volume)],
        marker_color=['#e74c3c', '#3498db', '#2ecc71'],
        text=[f'{result.cut_volume:,.0f}', f'{result.fill_volume:,.0f}', f'{result.net_volume:,.0f}'],
        textposition='auto',
    ))

    fig.update_layout(
        title='Volume Summary',
        yaxis_title='Volume (cubic units)',
        showlegend=False,
        height=350,
    )

    return fig


def create_area_chart(result):
    """Create pie chart of areas."""
    fig = go.Figure(data=[go.Pie(
        labels=['Cut Area', 'Fill Area'],
        values=[result.cut_area, result.fill_area],
        marker_colors=['#e74c3c', '#3498db'],
        hole=0.4,
    )])

    fig.update_layout(
        title='Area Distribution',
        height=350,
    )

    return fig


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🏗️ LIDAR Excavation Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze point clouds and calculate cut/fill volumes for construction planning</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Input")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Point Cloud",
            type=['las', 'laz', 'xyz', 'txt', 'ply'],
            help="Supported formats: LAS, LAZ, XYZ, PLY"
        )

        # Or use demo file
        use_demo = st.checkbox("Use demo terrain", value=not uploaded_file)

        st.divider()

        st.header("⚙️ Parameters")

        resolution = st.slider(
            "Grid Resolution",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Cell size in meters. Smaller = more detail, slower processing."
        )

        ground_only = st.checkbox(
            "Ground points only",
            value=True,
            help="Filter to ground-classified points (class 2)"
        )

        st.divider()

        st.header("📐 Work Area")

        area_method = st.radio(
            "Define area by:",
            ["Rectangle", "Polygon coordinates"],
            help="Rectangle is easier, polygon for complex shapes"
        )

        if area_method == "Rectangle":
            col1, col2 = st.columns(2)
            with col1:
                rect_x = st.number_input("X origin", value=10.0)
                rect_w = st.number_input("Width", value=30.0, min_value=1.0)
            with col2:
                rect_y = st.number_input("Y origin", value=10.0)
                rect_h = st.number_input("Height", value=30.0, min_value=1.0)
        else:
            polygon_json = st.text_area(
                "Polygon (JSON)",
                value="[[10,10],[40,10],[40,40],[10,40]]",
                help="Array of [x,y] coordinates"
            )

        st.divider()

        st.header("🎯 Grading Design")

        design_mode = st.radio(
            "Elevation mode:",
            ["Manual", "Optimize"],
        )

        if design_mode == "Manual":
            target_elevation = st.number_input(
                "Target Elevation",
                value=100.0,
                help="Desired finished grade elevation"
            )
        else:
            optimize_constraint = st.selectbox(
                "Optimization goal:",
                ["balance", "min_cut", "min_fill", "min_total"],
                format_func=lambda x: {
                    "balance": "Balance cut & fill",
                    "min_cut": "Minimize excavation",
                    "min_fill": "Minimize fill needed",
                    "min_total": "Minimize total earthwork",
                }[x]
            )

        st.divider()

        st.header("📊 Soil Factors")

        swell_factor = st.slider(
            "Swell Factor",
            min_value=1.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="Expansion factor for excavated material"
        )

        shrink_factor = st.slider(
            "Shrink Factor",
            min_value=0.7,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="Compaction factor for fill material"
        )

        st.divider()

        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

    # Main content area
    if analyze_button or 'result' in st.session_state:
        try:
            with st.spinner("Loading point cloud..."):
                # Load point cloud
                if uploaded_file:
                    pc = load_point_cloud(uploaded_file)
                elif use_demo:
                    from .io.point_cloud import PointCloudLoader
                    demo_path = Path(__file__).parent.parent.parent / "demo_terrain.las"
                    if demo_path.exists():
                        pc = PointCloudLoader.load(str(demo_path))
                    else:
                        # Generate sample terrain
                        from .io.point_cloud import generate_sample_terrain
                        pc = generate_sample_terrain(
                            size=(50.0, 50.0),
                            resolution=1.0,
                            base_elevation=100.0,
                            seed=42,
                        )
                else:
                    st.error("Please upload a point cloud file or use the demo terrain.")
                    return

            # Point cloud info
            with st.expander("📋 Point Cloud Info", expanded=False):
                bounds_min, bounds_max = pc.bounds
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Points", f"{pc.num_points:,}")
                with col2:
                    st.metric("X Range", f"{bounds_max[0] - bounds_min[0]:.1f}")
                with col3:
                    st.metric("Z Range", f"{bounds_max[2] - bounds_min[2]:.1f}")

            with st.spinner("Creating terrain grid..."):
                terrain = create_terrain_grid(pc, resolution, ground_only)

            # Create work polygon
            from shapely.geometry import Polygon as ShapelyPolygon, box

            if area_method == "Rectangle":
                work_polygon = box(rect_x, rect_y, rect_x + rect_w, rect_y + rect_h)
            else:
                coords = json.loads(polygon_json)
                work_polygon = ShapelyPolygon(coords)

            # Calculate volumes
            from .core.volume import VolumeCalculator

            calculator = VolumeCalculator(
                terrain,
                swell_factor=swell_factor,
                shrink_factor=shrink_factor,
            )

            with st.spinner("Calculating earthwork..."):
                if design_mode == "Manual":
                    result = calculator.calculate_flat(work_polygon, target_elevation)
                    optimal_elev = target_elevation
                else:
                    optimal_elev, result = calculator.find_optimal_elevation(
                        work_polygon,
                        constraint=optimize_constraint
                    )

            # Store in session state
            st.session_state['result'] = result
            st.session_state['terrain'] = terrain
            st.session_state['optimal_elev'] = optimal_elev

            # Results section
            st.header("📊 Analysis Results")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Cut Volume",
                    f"{result.cut_volume:,.0f}",
                    help="Material to be excavated"
                )

            with col2:
                st.metric(
                    "Fill Volume",
                    f"{result.fill_volume:,.0f}",
                    help="Material needed for fill"
                )

            with col3:
                st.metric(
                    "Net Volume",
                    f"{result.net_volume:,.0f}",
                    delta=f"{'Export' if result.net_volume > 0 else 'Import'}",
                    help="Positive = excess material, Negative = need import"
                )

            with col4:
                st.metric(
                    "Target Elevation",
                    f"{optimal_elev:.2f}",
                    help="Design grade elevation"
                )

            st.divider()

            # Visualizations
            tab1, tab2, tab3 = st.tabs(["🗺️ 3D Terrain", "📈 Cut/Fill Map", "📊 Charts"])

            with tab1:
                fig_3d = create_3d_terrain_plot(terrain)
                st.plotly_chart(fig_3d, use_container_width=True)

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    fig_terrain = create_2d_terrain_plot(terrain)
                    st.plotly_chart(fig_terrain, use_container_width=True)
                with col2:
                    result.target_elevation = optimal_elev
                    fig_cutfill = create_cut_fill_plot(result, terrain)
                    st.plotly_chart(fig_cutfill, use_container_width=True)

            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    fig_vol = create_volume_chart(result)
                    st.plotly_chart(fig_vol, use_container_width=True)
                with col2:
                    fig_area = create_area_chart(result)
                    st.plotly_chart(fig_area, use_container_width=True)

            # Detailed stats
            with st.expander("📋 Detailed Statistics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Volumes")
                    st.write(f"- Cut Volume: {result.cut_volume:,.2f} cubic units")
                    st.write(f"- Fill Volume: {result.fill_volume:,.2f} cubic units")
                    st.write(f"- Net Volume: {result.net_volume:,.2f} cubic units")
                    st.write(f"- Adjusted Cut (with swell): {result.adjusted_cut_volume:,.2f}")
                    st.write(f"- Adjusted Fill (with shrink): {result.adjusted_fill_volume:,.2f}")

                with col2:
                    st.subheader("Areas")
                    st.write(f"- Total Area: {result.total_area:,.2f} sq units")
                    st.write(f"- Cut Area: {result.cut_area:,.2f} sq units ({result.cut_area/result.total_area*100:.1f}%)")
                    st.write(f"- Fill Area: {result.fill_area:,.2f} sq units ({result.fill_area/result.total_area*100:.1f}%)")

                st.subheader("Terrain Statistics")
                stats = terrain.statistics()
                st.write(f"- Grid Size: {terrain.rows} x {terrain.cols} cells")
                st.write(f"- Resolution: {terrain.resolution} units/cell")
                st.write(f"- Elevation Range: {stats['min_elevation']:.2f} to {stats['max_elevation']:.2f}")
                st.write(f"- Mean Elevation: {stats['mean_elevation']:.2f}")

            # Export section
            st.header("💾 Export Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                # JSON export
                json_data = json.dumps(result.to_dict(), indent=2)
                st.download_button(
                    "📄 Download JSON",
                    data=json_data,
                    file_name="excavation_results.json",
                    mime="application/json",
                )

            with col2:
                # CSV export
                from .io.exporters import export_cell_details_csv
                csv_buffer = BytesIO()

                # Create CSV content
                csv_lines = ["x,y,existing_elevation,design_elevation,cut_depth,fill_depth,cut_volume,fill_volume"]
                if hasattr(result, 'cell_details') and result.cell_details:
                    for cell in result.cell_details:
                        csv_lines.append(
                            f"{cell.get('x', 0)},{cell.get('y', 0)},"
                            f"{cell.get('existing_elevation', 0)},{cell.get('design_elevation', 0)},"
                            f"{cell.get('cut_depth', 0)},{cell.get('fill_depth', 0)},"
                            f"{cell.get('cut_volume', 0)},{cell.get('fill_volume', 0)}"
                        )
                csv_content = "\n".join(csv_lines)

                st.download_button(
                    "📊 Download CSV",
                    data=csv_content,
                    file_name="cell_details.csv",
                    mime="text/csv",
                )

            with col3:
                # Summary report
                report = f"""LIDAR EXCAVATION ANALYSIS REPORT
================================

Target Elevation: {optimal_elev:.2f}

VOLUMES
-------
Cut Volume:     {result.cut_volume:>12,.2f} cubic units
Fill Volume:    {result.fill_volume:>12,.2f} cubic units
Net Volume:     {result.net_volume:>12,.2f} cubic units

AREAS
-----
Total Area:     {result.total_area:>12,.2f} sq units
Cut Area:       {result.cut_area:>12,.2f} sq units
Fill Area:      {result.fill_area:>12,.2f} sq units

SOIL ADJUSTMENTS
----------------
Swell Factor:   {swell_factor}
Shrink Factor:  {shrink_factor}
Adjusted Cut:   {result.adjusted_cut_volume:>12,.2f} cubic units
Adjusted Fill:  {result.adjusted_fill_volume:>12,.2f} cubic units
"""
                st.download_button(
                    "📝 Download Report",
                    data=report,
                    file_name="excavation_report.txt",
                    mime="text/plain",
                )

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.exception(e)

    else:
        # Welcome / instructions when no analysis run yet
        st.info("👈 Configure parameters in the sidebar and click **Analyze** to begin.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📖 How to Use")
            st.markdown("""
            1. **Upload** a point cloud file (LAS, LAZ, XYZ) or use the demo terrain
            2. **Set resolution** - smaller values = more detail
            3. **Define work area** - rectangle or polygon coordinates
            4. **Choose elevation** - manual or let the tool optimize
            5. **Click Analyze** to calculate cut/fill volumes
            """)

        with col2:
            st.subheader("📋 Supported Formats")
            st.markdown("""
            - **LAS/LAZ**: Industry standard LIDAR format
            - **XYZ**: Simple text format (X Y Z per line)
            - **PLY**: Polygon file format

            For best results, use ground-classified LAS files.
            """)


if __name__ == "__main__":
    main()
