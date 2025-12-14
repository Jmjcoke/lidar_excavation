"""
Point Cloud Loading Module

Handles loading LIDAR data from various formats (LAS, LAZ, XYZ, etc.)
and provides a unified PointCloud data structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Iterator
import numpy as np

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False


def _extract_crs_from_las(las) -> Optional[str]:
    """
    Extract CRS from LAS file VLRs (Variable Length Records).

    LAS files store CRS in VLRs using:
    - WKT strings (user_id="LASF_WKT", record_id=1) - LAS 1.4+
    - Legacy WKT (user_id="LASF_Projection", record_id=2112) - LAS 1.0-1.3
    - GeoTIFF keys (user_id="LASF_Projection", record_id=34735)

    Args:
        las: laspy LasData object

    Returns:
        CRS as WKT string or EPSG code string (e.g., "EPSG:32610"), or None
    """
    if not hasattr(las, 'vlrs') or not las.vlrs:
        return None

    # Try WKT-based VLRs first (more reliable)
    for vlr in las.vlrs:
        # OGC WKT (LAS 1.4+)
        if vlr.user_id == "LASF_WKT" and vlr.record_id == 1:
            try:
                wkt = vlr.record_data.decode('utf-8').rstrip('\x00')
                if wkt.strip():
                    return wkt
            except (AttributeError, UnicodeDecodeError):
                pass

        # Legacy WKT (LAS 1.0-1.3)
        if vlr.user_id == "LASF_Projection" and vlr.record_id == 2112:
            try:
                wkt = vlr.record_data.decode('utf-8').rstrip('\x00')
                if wkt.strip():
                    return wkt
            except (AttributeError, UnicodeDecodeError):
                pass

    # Try GeoTIFF keys (more complex parsing)
    for vlr in las.vlrs:
        if vlr.user_id == "LASF_Projection" and vlr.record_id == 34735:
            # GeoKeyDirectoryTag - contains key/value pairs
            # Parse to find ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048)
            try:
                import struct
                data = vlr.record_data
                if len(data) >= 8:
                    # Header: KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys
                    num_keys = struct.unpack('<H', data[6:8])[0]
                    offset = 8
                    for _ in range(num_keys):
                        if offset + 8 > len(data):
                            break
                        key_id, tiff_tag, count, value = struct.unpack(
                            '<HHHH', data[offset:offset+8]
                        )
                        # ProjectedCSTypeGeoKey = 3072, GeographicTypeGeoKey = 2048
                        if key_id == 3072 and tiff_tag == 0:  # Short value inline
                            return f"EPSG:{value}"
                        if key_id == 2048 and tiff_tag == 0:
                            return f"EPSG:{value}"
                        offset += 8
            except Exception:
                pass

    return None


@dataclass
class PointCloud:
    """
    Unified point cloud data structure.

    Attributes:
        xyz: Nx3 array of point coordinates
        classification: Optional Nx1 array of point classifications
            (2 = ground, 6 = building, etc. per ASPRS LAS spec)
        intensity: Optional Nx1 array of return intensity values
        rgb: Optional Nx3 array of RGB colors (0-65535 or 0-255)
        crs: Coordinate reference system (EPSG code or WKT)
    """
    xyz: np.ndarray
    classification: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    crs: Optional[str] = None
    _bounds: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate data shapes."""
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be Nx3 array, got shape {self.xyz.shape}")

        n_points = len(self.xyz)

        if self.classification is not None and len(self.classification) != n_points:
            raise ValueError("classification length must match xyz")
        if self.intensity is not None and len(self.intensity) != n_points:
            raise ValueError("intensity length must match xyz")
        if self.rgb is not None and len(self.rgb) != n_points:
            raise ValueError("rgb length must match xyz")

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (min_xyz, max_xyz) bounding box."""
        if self._bounds is None:
            self._bounds = (
                np.min(self.xyz, axis=0),
                np.max(self.xyz, axis=0)
            )
        return self._bounds

    @property
    def num_points(self) -> int:
        """Total number of points."""
        return len(self.xyz)

    @property
    def x(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz[:, 2]

    def filter_by_classification(self, classes: list[int]) -> PointCloud:
        """
        Return a new PointCloud containing only points with specified classifications.

        Args:
            classes: List of classification codes to keep (e.g., [2] for ground only)

        Returns:
            New filtered PointCloud
        """
        if self.classification is None:
            raise ValueError("Point cloud has no classification data")

        mask = np.isin(self.classification, classes)
        return self._apply_mask(mask)

    def filter_by_bounds(
        self,
        min_x: float = -np.inf,
        max_x: float = np.inf,
        min_y: float = -np.inf,
        max_y: float = np.inf,
        min_z: float = -np.inf,
        max_z: float = np.inf,
    ) -> PointCloud:
        """Filter points by spatial bounds."""
        mask = (
            (self.xyz[:, 0] >= min_x) & (self.xyz[:, 0] <= max_x) &
            (self.xyz[:, 1] >= min_y) & (self.xyz[:, 1] <= max_y) &
            (self.xyz[:, 2] >= min_z) & (self.xyz[:, 2] <= max_z)
        )
        return self._apply_mask(mask)

    def _apply_mask(self, mask: np.ndarray) -> PointCloud:
        """Apply boolean mask to create filtered point cloud."""
        return PointCloud(
            xyz=self.xyz[mask].copy(),
            classification=self.classification[mask].copy() if self.classification is not None else None,
            intensity=self.intensity[mask].copy() if self.intensity is not None else None,
            rgb=self.rgb[mask].copy() if self.rgb is not None else None,
            crs=self.crs,
        )

    def subsample(self, factor: int = 10) -> PointCloud:
        """Return every Nth point (for quick visualization/testing)."""
        return self._apply_mask(np.arange(len(self.xyz)) % factor == 0)

    def remove_statistical_outliers(
        self,
        k_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> PointCloud:
        """
        Remove statistical outliers based on mean distance to k neighbors.

        Points with mean neighbor distance > (global_mean + std_ratio * global_std)
        are considered outliers and removed.
        """
        from scipy.spatial import KDTree

        tree = KDTree(self.xyz)
        distances, _ = tree.query(self.xyz, k=k_neighbors + 1)

        # Skip first column (distance to self = 0)
        mean_distances = np.mean(distances[:, 1:], axis=1)

        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_ratio * global_std

        mask = mean_distances <= threshold
        return self._apply_mask(mask)


class PointCloudLoader:
    """
    Factory for loading point clouds from various file formats.

    Supported formats:
        - LAS/LAZ (requires laspy)
        - XYZ (plain text: x y z per line)
        - PLY (ASCII point clouds)
    """

    # ASPRS LAS Classification codes
    CLASS_UNCLASSIFIED = 1
    CLASS_GROUND = 2
    CLASS_LOW_VEGETATION = 3
    CLASS_MEDIUM_VEGETATION = 4
    CLASS_HIGH_VEGETATION = 5
    CLASS_BUILDING = 6
    CLASS_NOISE = 7
    CLASS_WATER = 9

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> PointCloud:
        """
        Load point cloud from file, auto-detecting format.

        Args:
            filepath: Path to point cloud file
            **kwargs: Format-specific options

        Returns:
            PointCloud instance
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        loaders = {
            '.las': cls._load_las,
            '.laz': cls._load_las,
            '.xyz': cls._load_xyz,
            '.txt': cls._load_xyz,
            '.ply': cls._load_ply,
        }

        if suffix not in loaders:
            raise ValueError(f"Unsupported format: {suffix}")

        return loaders[suffix](filepath, **kwargs)

    @classmethod
    def _load_las(cls, filepath: Path, **kwargs) -> PointCloud:
        """Load LAS/LAZ file using laspy."""
        suffix = filepath.suffix.lower()

        if not HAS_LASPY:
            if suffix == '.laz':
                raise ImportError(
                    "laspy is required to load LAZ files.\n"
                    "Install with: pip install laspy lazrs\n\n"
                    "Note for Windows users: The 'lazrs' package provides LAZ "
                    "decompression. If installation fails, try:\n"
                    "  pip install laspy[lazrs]\n"
                    "Or use LAS files (uncompressed) as an alternative."
                )
            else:
                raise ImportError(
                    "laspy is required to load LAS files. "
                    "Install with: pip install laspy"
                )

        # Try to load the file with better error handling for LAZ
        try:
            with laspy.open(filepath) as reader:
                las = reader.read()
        except Exception as e:
            error_msg = str(e).lower()
            if suffix == '.laz' and ('lazrs' in error_msg or 'laz' in error_msg or 'decompress' in error_msg):
                raise ImportError(
                    f"Failed to decompress LAZ file: {e}\n\n"
                    "LAZ decompression requires the 'lazrs' package.\n"
                    "Install with: pip install lazrs\n\n"
                    "Windows users: If lazrs fails to install, try:\n"
                    "  1. pip install laspy[lazrs]\n"
                    "  2. Use LAS files instead of LAZ\n"
                    "  3. Convert LAZ to LAS using LAStools or CloudCompare"
                ) from e
            raise

        # Extract coordinates (scaled)
        xyz = np.column_stack([
            las.x,
            las.y,
            las.z
        ]).astype(np.float64)

        # Classification (if available)
        classification = None
        if hasattr(las, 'classification'):
            classification = np.array(las.classification, dtype=np.uint8)

        # Intensity (if available)
        intensity = None
        if hasattr(las, 'intensity'):
            intensity = np.array(las.intensity, dtype=np.uint16)

        # RGB (if available)
        rgb = None
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            rgb = np.column_stack([
                las.red,
                las.green,
                las.blue
            ]).astype(np.uint16)

        # CRS from VLRs (if available)
        crs = _extract_crs_from_las(las)

        return PointCloud(
            xyz=xyz,
            classification=classification,
            intensity=intensity,
            rgb=rgb,
            crs=crs,
        )

    @classmethod
    def _load_xyz(
        cls,
        filepath: Path,
        delimiter: str = None,
        skip_header: int = 0,
        **kwargs
    ) -> PointCloud:
        """
        Load XYZ text file.

        Expected format: x y z [classification] [intensity] per line
        """
        data = np.loadtxt(
            filepath,
            delimiter=delimiter,
            skiprows=skip_header,
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 3:
            raise ValueError("XYZ file must have at least 3 columns")

        xyz = data[:, :3].astype(np.float64)

        classification = None
        if data.shape[1] >= 4:
            classification = data[:, 3].astype(np.uint8)

        intensity = None
        if data.shape[1] >= 5:
            intensity = data[:, 4].astype(np.uint16)

        return PointCloud(xyz=xyz, classification=classification, intensity=intensity)

    @classmethod
    def _load_ply(cls, filepath: Path, **kwargs) -> PointCloud:
        """Load PLY file (ASCII format only for now)."""
        with open(filepath, 'r') as f:
            # Parse header
            line = f.readline().strip()
            if line != 'ply':
                raise ValueError("Not a valid PLY file")

            n_vertices = 0
            properties = []
            in_header = True

            while in_header:
                line = f.readline().strip()
                if line.startswith('element vertex'):
                    n_vertices = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    properties.append(parts[-1])  # property name
                elif line == 'end_header':
                    in_header = False

            # Find column indices
            x_idx = properties.index('x') if 'x' in properties else 0
            y_idx = properties.index('y') if 'y' in properties else 1
            z_idx = properties.index('z') if 'z' in properties else 2

            # Read data
            data = np.loadtxt(f, max_rows=n_vertices)

            if data.ndim == 1:
                data = data.reshape(1, -1)

            xyz = np.column_stack([
                data[:, x_idx],
                data[:, y_idx],
                data[:, z_idx]
            ]).astype(np.float64)

            # Check for RGB
            rgb = None
            if 'red' in properties and 'green' in properties and 'blue' in properties:
                rgb = np.column_stack([
                    data[:, properties.index('red')],
                    data[:, properties.index('green')],
                    data[:, properties.index('blue')]
                ]).astype(np.uint16)

            return PointCloud(xyz=xyz, rgb=rgb)

    @classmethod
    def load_chunked(
        cls,
        filepath: str | Path,
        chunk_size: int = 1_000_000
    ) -> Iterator[PointCloud]:
        """
        Load large LAS/LAZ files in chunks to manage memory.

        Args:
            filepath: Path to LAS/LAZ file
            chunk_size: Number of points per chunk

        Yields:
            PointCloud instances for each chunk
        """
        if not HAS_LASPY:
            raise ImportError("laspy required for chunked loading")

        filepath = Path(filepath)

        with laspy.open(filepath) as reader:
            for points in reader.chunk_iterator(chunk_size):
                xyz = np.column_stack([points.x, points.y, points.z]).astype(np.float64)

                classification = None
                if hasattr(points, 'classification'):
                    classification = np.array(points.classification, dtype=np.uint8)

                yield PointCloud(xyz=xyz, classification=classification)


def generate_sample_terrain(
    size: Tuple[float, float] = (100.0, 100.0),
    resolution: float = 1.0,
    base_elevation: float = 100.0,
    noise_scale: float = 5.0,
    hill_height: float = 10.0,
    seed: int = 42,
) -> PointCloud:
    """
    Generate synthetic terrain point cloud for testing.

    Creates a terrain with gentle hills and random noise,
    useful for testing without real LIDAR data.

    Args:
        size: (width, height) in meters
        resolution: Point spacing in meters
        base_elevation: Base elevation value
        noise_scale: Amount of random noise
        hill_height: Maximum hill height
        seed: Random seed for reproducibility

    Returns:
        PointCloud with synthetic terrain
    """
    np.random.seed(seed)

    width, height = size
    x = np.arange(0, width, resolution)
    y = np.arange(0, height, resolution)
    xx, yy = np.meshgrid(x, y)

    # Create terrain with hills using sin waves
    zz = base_elevation + (
        hill_height * np.sin(xx / 20) * np.cos(yy / 25) +
        hill_height * 0.5 * np.sin(xx / 10 + yy / 15) +
        noise_scale * np.random.randn(*xx.shape)
    )

    # Flatten to point arrays
    xyz = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        zz.ravel()
    ])

    # All points classified as ground
    classification = np.full(len(xyz), PointCloudLoader.CLASS_GROUND, dtype=np.uint8)

    return PointCloud(xyz=xyz, classification=classification)
