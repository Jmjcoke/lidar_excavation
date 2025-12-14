# LIDAR Excavation

A Python library and CLI tool for analyzing LIDAR point clouds to calculate cut/fill volumes for construction excavation planning.

## Features

- **Point Cloud I/O**: Load LAS/LAZ files (via laspy), XYZ text files, and PLY files with automatic CRS extraction
- **Terrain Grid Creation**: Generate digital elevation models from point clouds with configurable resolution and interpolation methods
- **Volume Calculation**: Calculate cut and fill volumes for flat and sloped grading designs
- **Grading Analysis**: Building pad and parking lot design tools with drainage analysis
- **Optimization**: Find optimal grading elevations to balance cut/fill or minimize total earthwork
- **Visualization**: 2D/3D terrain plots, cut/fill heatmaps, and cross-section views
- **Export Formats**: JSON summaries, CSV cell details, GeoJSON zones, and GeoTIFF DEMs

## Installation

```bash
# Clone the repository
git clone https://github.com/Jmjcoke/lidar_excavation.git
cd lidar_excavation

# Install in development mode
pip install -e .

# For full features (GeoTIFF export, 3D visualization)
pip install -e ".[full]"

# For development (testing, linting)
pip install -e ".[dev]"
```

### Dependencies

- Python 3.9+
- numpy, scipy, shapely
- laspy with lazrs (for LAS/LAZ files)
- matplotlib (for visualization)
- click (for CLI)

Optional:
- rasterio (for GeoTIFF export)
- open3d (for 3D visualization)

## Quick Start

### Web Interface (Recommended)

Launch the browser-based interface:

```bash
streamlit run src/lidar_excavation/app.py
```

Or after installation:

```bash
lidar-excavation-web
```

Then open http://localhost:8501 in your browser.

![Web Interface](examples/web_interface.png)

### Command Line

```bash
# View point cloud info
lidar-excavation info terrain.las

# Calculate cut/fill volumes for a rectangular area
lidar-excavation analyze terrain.las --elevation 100.0 --rect "50,50,100,75"

# Calculate with a custom polygon
lidar-excavation analyze terrain.las --elevation 100.0 \
    --polygon "[[0,0],[100,0],[100,50],[0,50]]"

# Find optimal grading elevation
lidar-excavation optimize terrain.las \
    --polygon "[[0,0],[100,0],[100,50],[0,50]]" \
    --constraint balance

# Convert to DEM GeoTIFF
lidar-excavation to-dem terrain.las -o terrain.tif -r 0.5

# Generate sample terrain for testing
lidar-excavation generate-sample -o sample.xyz --size 200,200
```

### Python API

```python
from lidar_excavation.io import PointCloudLoader
from lidar_excavation.core import TerrainGrid, VolumeCalculator
from shapely.geometry import box

# Load point cloud
pc = PointCloudLoader.load("terrain.las")

# Create terrain grid (1m resolution, ground points only)
terrain = TerrainGrid.from_point_cloud(pc, resolution=1.0, ground_only=True)

# Define work area
work_area = box(50, 50, 150, 125)  # 100m x 75m rectangle

# Calculate volumes
calculator = VolumeCalculator(terrain)
result = calculator.calculate_flat(work_area, target_elevation=100.0)

print(f"Cut volume:  {result.cut_volume:,.1f} cubic units")
print(f"Fill volume: {result.fill_volume:,.1f} cubic units")
print(f"Net volume:  {result.net_volume:,.1f} cubic units")

# Find optimal elevation for balanced earthwork
optimal_elev, result = calculator.find_optimal_elevation(work_area, constraint="balance")
print(f"Optimal elevation: {optimal_elev:.2f}")
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `info` | Display point cloud file information |
| `analyze` | Calculate cut/fill volumes for a work area |
| `optimize` | Find optimal grading elevation |
| `generate-sample` | Create synthetic terrain for testing |
| `to-dem` | Convert point cloud to GeoTIFF DEM |

### Analyze Options

```
--elevation, -e    Target grading elevation (required)
--polygon, -p      Work area as JSON coordinates
--rect             Work area as "x,y,width,height"
--resolution, -r   Grid cell size (default: 1.0)
--output, -o       Save results to JSON file
--export-csv       Export cell details to CSV
--export-geojson   Export cut/fill zones to GeoJSON
--plot             Show visualization
--swell-factor     Soil swell factor for cut volumes
--shrink-factor    Soil shrink factor for fill volumes
```

### Optimization Constraints

| Constraint | Description |
|------------|-------------|
| `balance` | Minimize difference between cut and fill |
| `min_cut` | Minimize excavation volume |
| `min_fill` | Minimize fill material needed |
| `min_total` | Minimize total earthwork (cut + fill) |

## Export Formats

### JSON Summary
```json
{
  "cut_volume": 1234.5,
  "fill_volume": 987.6,
  "net_volume": 246.9,
  "total_area": 7500.0,
  "cut_area": 4200.0,
  "fill_area": 3300.0
}
```

### CSV Cell Details
Cell-by-cell breakdown with coordinates, existing elevation, design elevation, and cut/fill depths.

### GeoJSON Zones
Polygons for cut and fill zones, suitable for GIS applications.

### GeoTIFF DEM
Raster elevation model with embedded CRS for use in GIS software.

## Project Structure

```
lidar_excavation/
├── src/lidar_excavation/
│   ├── core/           # Core algorithms
│   │   ├── terrain.py      # Terrain grid and DEM
│   │   ├── volume.py       # Cut/fill calculations
│   │   └── validation.py   # Input validation
│   ├── io/             # File I/O
│   │   ├── point_cloud.py  # Point cloud loading
│   │   └── exporters.py    # Export utilities
│   ├── analysis/       # Analysis tools
│   │   └── grading.py      # Building pad, parking lot
│   ├── utils/          # Utilities
│   │   └── visualization.py
│   └── cli.py          # Command line interface
├── tests/              # Test suite
├── examples/           # Example scripts
└── demo_terrain.las    # Sample LAS file
```

## Testing

```bash
# Run all tests
pytest

# Run fast tests only
pytest -m "not slow"

# Run with coverage
pytest --cov=lidar_excavation
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
