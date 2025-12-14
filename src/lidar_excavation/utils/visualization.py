"""
Visualization Utilities

Plotting functions for terrain, earthwork analysis, and cross-sections.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, TYPE_CHECKING
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Polygon as MplPolygon
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from ..core.terrain import TerrainGrid
    from ..core.volume import EarthworkResult


def require_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")


# Custom colormap for cut/fill (red = cut, blue = fill)
CUT_FILL_COLORS = [
    (0.0, (0.2, 0.2, 0.8)),   # Deep blue (fill)
    (0.4, (0.6, 0.8, 1.0)),   # Light blue
    (0.5, (0.95, 0.95, 0.95)), # White (no change)
    (0.6, (1.0, 0.8, 0.6)),   # Light red/orange
    (1.0, (0.8, 0.2, 0.2)),   # Deep red (cut)
]


def get_cut_fill_cmap():
    """Get the cut/fill colormap."""
    require_matplotlib()
    return LinearSegmentedColormap.from_list("cut_fill", CUT_FILL_COLORS)


def plot_terrain(
    terrain: 'TerrainGrid',
    ax: Optional[plt.Axes] = None,
    title: str = "Terrain Elevation",
    cmap: str = "terrain",
    show_contours: bool = True,
    contour_interval: float = 1.0,
    polygon: Optional['Polygon'] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot terrain elevation as a 2D heatmap with optional contours.

    Args:
        terrain: TerrainGrid to plot
        ax: Optional matplotlib axes (creates new figure if None)
        title: Plot title
        cmap: Colormap name
        show_contours: Whether to show contour lines
        contour_interval: Elevation interval for contour lines
        polygon: Optional polygon to overlay
        figsize: Figure size if creating new figure

    Returns:
        matplotlib Figure
    """
    require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Mask nodata values
    data = np.ma.masked_equal(terrain.elevations, terrain.nodata)

    # Calculate extent
    bounds = terrain.bounds
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    # Plot heatmap
    im = ax.imshow(
        data,
        extent=extent,
        origin='lower',
        cmap=cmap,
        aspect='equal',
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Elevation')

    # Add contours
    if show_contours:
        valid_data = data.compressed()
        if len(valid_data) > 0:
            min_elev = np.floor(valid_data.min() / contour_interval) * contour_interval
            max_elev = np.ceil(valid_data.max() / contour_interval) * contour_interval
            levels = np.arange(min_elev, max_elev + contour_interval, contour_interval)

            if len(levels) > 1:
                cs = ax.contour(
                    data,
                    levels=levels,
                    extent=extent,
                    origin='lower',
                    colors='black',
                    linewidths=0.5,
                    alpha=0.5,
                )
                ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Overlay polygon
    if polygon is not None:
        try:
            from shapely.geometry import Polygon
            if isinstance(polygon, Polygon):
                x, y = polygon.exterior.xy
                ax.plot(x, y, 'r-', linewidth=2, label='Design Area')
                ax.legend()
        except ImportError:
            pass

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    return fig


def plot_terrain_3d(
    terrain: 'TerrainGrid',
    ax: Optional[Axes3D] = None,
    title: str = "3D Terrain",
    cmap: str = "terrain",
    vertical_exaggeration: float = 1.0,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot terrain as a 3D surface.

    Args:
        terrain: TerrainGrid to plot
        ax: Optional 3D axes
        title: Plot title
        cmap: Colormap name
        vertical_exaggeration: Z-axis scaling factor
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    require_matplotlib()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Create coordinate grids
    x = terrain.x_coords
    y = terrain.y_coords
    X, Y = np.meshgrid(x, y)

    # Mask and scale elevations
    Z = np.ma.masked_equal(terrain.elevations, terrain.nodata)
    Z = Z * vertical_exaggeration

    # Plot surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.set_title(title)

    plt.colorbar(surf, ax=ax, shrink=0.5, label='Elevation')

    return fig


def plot_earthwork(
    terrain: 'TerrainGrid',
    result: 'EarthworkResult',
    ax: Optional[plt.Axes] = None,
    title: str = "Cut/Fill Analysis",
    figsize: Tuple[int, int] = (12, 8),
    max_depth: Optional[float] = None,
) -> plt.Figure:
    """
    Plot cut/fill heatmap from earthwork analysis.

    Red = Cut (excavation needed)
    Blue = Fill (material needed)

    Args:
        terrain: TerrainGrid
        result: EarthworkResult from volume calculation
        ax: Optional axes
        title: Plot title
        figsize: Figure size
        max_depth: Optional max depth for color scaling (symmetric)

    Returns:
        matplotlib Figure
    """
    require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create cut/fill grid
    net_grid = np.zeros(terrain.shape)
    for cell in result.cell_details:
        net_grid[cell.row, cell.col] = cell.cut_depth - cell.fill_depth

    # Mask areas outside design
    mask = np.ones(terrain.shape, dtype=bool)
    for cell in result.cell_details:
        mask[cell.row, cell.col] = False
    net_grid = np.ma.masked_array(net_grid, mask)

    # Calculate extent
    bounds = terrain.bounds
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    # Determine color scale
    if max_depth is None:
        max_depth = max(result.max_cut_depth, result.max_fill_depth)

    if max_depth == 0:
        max_depth = 1.0

    # Plot with diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=-max_depth, vcenter=0, vmax=max_depth)

    im = ax.imshow(
        net_grid,
        extent=extent,
        origin='lower',
        cmap=get_cut_fill_cmap(),
        norm=norm,
        aspect='equal',
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cut (+) / Fill (-) Depth')

    # Add summary text
    summary_text = (
        f"Cut: {result.cut_volume:,.0f} cu units\n"
        f"Fill: {result.fill_volume:,.0f} cu units\n"
        f"Net: {result.net_volume:,.0f} cu units"
    )
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes,
        verticalalignment='top',
        fontfamily='monospace',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    return fig


def plot_cross_section(
    terrain: 'TerrainGrid',
    start: Tuple[float, float],
    end: Tuple[float, float],
    target_elevation: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Cross Section",
    num_points: int = 100,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot a cross-section profile along a line.

    Args:
        terrain: TerrainGrid
        start: (x, y) start point
        end: (x, y) end point
        target_elevation: Optional target elevation to show
        ax: Optional axes
        title: Plot title
        num_points: Number of sample points along line
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Sample points along line
    x_coords = np.linspace(start[0], end[0], num_points)
    y_coords = np.linspace(start[1], end[1], num_points)

    # Calculate distance along profile
    distances = np.sqrt(
        (x_coords - start[0])**2 +
        (y_coords - start[1])**2
    )

    # Get elevations
    elevations = []
    for x, y in zip(x_coords, y_coords):
        elev = terrain.get_elevation_interpolated(x, y)
        elevations.append(elev if elev != terrain.nodata else np.nan)

    elevations = np.array(elevations)

    # Plot existing terrain
    ax.fill_between(
        distances,
        elevations,
        elevations.min() - 1,
        alpha=0.3,
        color='brown',
        label='Existing Ground',
    )
    ax.plot(distances, elevations, 'k-', linewidth=2, label='Existing Surface')

    # Plot target elevation
    if target_elevation is not None:
        ax.axhline(
            y=target_elevation,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Target ({target_elevation:.1f})',
        )

        # Fill cut and fill areas
        cut_mask = elevations > target_elevation
        fill_mask = elevations < target_elevation

        ax.fill_between(
            distances,
            elevations,
            target_elevation,
            where=cut_mask,
            alpha=0.5,
            color='red',
            label='Cut',
        )
        ax.fill_between(
            distances,
            elevations,
            target_elevation,
            where=fill_mask,
            alpha=0.5,
            color='blue',
            label='Fill',
        )

    ax.set_xlabel('Distance')
    ax.set_ylabel('Elevation')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    valid_elev = elevations[~np.isnan(elevations)]
    if len(valid_elev) > 0:
        y_range = valid_elev.max() - valid_elev.min()
        padding = max(y_range * 0.1, 1.0)
        ax.set_ylim(valid_elev.min() - padding, valid_elev.max() + padding)

    return fig


def create_report_figure(
    terrain: 'TerrainGrid',
    result: 'EarthworkResult',
    polygon: Optional['Polygon'] = None,
    cross_section_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    target_elevation: Optional[float] = None,
) -> plt.Figure:
    """
    Create a comprehensive report figure with multiple views.

    Args:
        terrain: TerrainGrid
        result: EarthworkResult
        polygon: Optional design polygon
        cross_section_line: Optional (start, end) for cross-section
        target_elevation: Optional target elevation

    Returns:
        matplotlib Figure with 4 subplots
    """
    require_matplotlib()

    fig = plt.figure(figsize=(16, 12))

    # 1. Terrain overview
    ax1 = fig.add_subplot(221)
    plot_terrain(terrain, ax=ax1, title="Existing Terrain", polygon=polygon)

    # 2. Cut/Fill heatmap
    ax2 = fig.add_subplot(222)
    plot_earthwork(terrain, result, ax=ax2, title="Cut/Fill Analysis")

    # 3. Cross-section (if provided)
    ax3 = fig.add_subplot(223)
    if cross_section_line:
        start, end = cross_section_line
        plot_cross_section(
            terrain, start, end,
            target_elevation=target_elevation,
            ax=ax3,
            title="Cross Section",
        )
    else:
        # Default: cross-section through center
        bounds = terrain.bounds
        center_y = (bounds[1] + bounds[3]) / 2
        start = (bounds[0], center_y)
        end = (bounds[2], center_y)
        plot_cross_section(
            terrain, start, end,
            target_elevation=target_elevation,
            ax=ax3,
            title="Cross Section (E-W through center)",
        )

    # 4. Summary statistics
    ax4 = fig.add_subplot(224)
    ax4.axis('off')

    summary_text = result.summary()
    ax4.text(
        0.1, 0.95, summary_text,
        transform=ax4.transAxes,
        verticalalignment='top',
        fontfamily='monospace',
        fontsize=10,
    )

    plt.tight_layout()
    return fig


def save_report(
    figure: plt.Figure,
    filepath: str,
    dpi: int = 150,
) -> None:
    """Save report figure to file."""
    figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Report saved to: {filepath}")
