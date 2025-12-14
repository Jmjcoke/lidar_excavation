"""
CLI command tests using Click's test runner.
"""

import pytest
import json

from lidar_excavation.cli import main


class TestCLIInfo:
    """Test 'info' command."""

    @pytest.mark.requires_laspy
    def test_info_demo_las(self, cli_runner, demo_las_path):
        """Test info command on demo LAS file."""
        result = cli_runner.invoke(main, ['info', str(demo_las_path)])

        assert result.exit_code == 0
        assert 'POINT CLOUD INFO' in result.output
        assert 'Points:' in result.output
        assert 'Bounds:' in result.output

    def test_info_missing_file(self, cli_runner):
        """Test info command with missing file."""
        result = cli_runner.invoke(main, ['info', 'nonexistent.las'])

        assert result.exit_code != 0


class TestCLIAnalyze:
    """Test 'analyze' command."""

    @pytest.mark.requires_laspy
    @pytest.mark.slow
    def test_analyze_with_rect(self, cli_runner, demo_las_path, tmp_output_dir):
        """Test analyze command with rectangle."""
        output_json = tmp_output_dir / "result.json"

        result = cli_runner.invoke(main, [
            'analyze', str(demo_las_path),
            '--elevation', '100.0',
            '--rect', '10,10,20,20',
            '--resolution', '2.0',
            '--output', str(output_json),
        ])

        assert result.exit_code == 0
        assert output_json.exists()

        # Verify JSON output
        with open(output_json) as f:
            data = json.load(f)
        assert 'cut_volume' in data
        assert 'fill_volume' in data

    def test_analyze_missing_polygon_and_rect(self, cli_runner, demo_las_path):
        """Test analyze fails without polygon or rect."""
        result = cli_runner.invoke(main, [
            'analyze', str(demo_las_path),
            '--elevation', '100.0',
        ])

        assert result.exit_code != 0
        assert 'Must specify either --polygon or --rect' in result.output

    @pytest.mark.requires_laspy
    @pytest.mark.slow
    def test_analyze_with_polygon_json(self, cli_runner, demo_las_path):
        """Test analyze with JSON polygon."""
        result = cli_runner.invoke(main, [
            'analyze', str(demo_las_path),
            '--elevation', '100.0',
            '--polygon', '[[10,10],[30,10],[30,30],[10,30]]',
            '--resolution', '2.0',
        ])

        assert result.exit_code == 0
        assert 'EARTHWORK ANALYSIS SUMMARY' in result.output

    def test_analyze_invalid_rect_format(self, cli_runner, demo_las_path):
        """Test analyze with invalid rectangle format."""
        result = cli_runner.invoke(main, [
            'analyze', str(demo_las_path),
            '--elevation', '100.0',
            '--rect', '10,10,20',  # Missing 4th value
        ])

        assert result.exit_code != 0
        assert 'exactly 4 values' in result.output

    def test_analyze_invalid_rect_values(self, cli_runner, demo_las_path):
        """Test analyze with invalid rectangle values."""
        result = cli_runner.invoke(main, [
            'analyze', str(demo_las_path),
            '--elevation', '100.0',
            '--rect', '10,10,-5,20',  # Negative width
        ])

        assert result.exit_code != 0
        assert 'must be positive' in result.output

    def test_analyze_invalid_resolution(self, cli_runner, demo_las_path):
        """Test analyze with invalid resolution."""
        result = cli_runner.invoke(main, [
            'analyze', str(demo_las_path),
            '--elevation', '100.0',
            '--rect', '10,10,20,20',
            '--resolution', '0',  # Invalid
        ])

        assert result.exit_code != 0
        assert 'Resolution must be positive' in result.output


class TestCLIGenerateSample:
    """Test 'generate-sample' command."""

    def test_generate_sample_xyz(self, cli_runner, tmp_output_dir):
        """Test generating sample terrain as XYZ."""
        output_file = tmp_output_dir / "sample.xyz"

        result = cli_runner.invoke(main, [
            'generate-sample',
            '--output', str(output_file),
            '--size', '20,20',
            '--resolution', '2.0',
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify file has content
        content = output_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) > 0

    @pytest.mark.requires_laspy
    def test_generate_sample_las(self, cli_runner, tmp_output_dir):
        """Test generating sample terrain as LAS."""
        output_file = tmp_output_dir / "sample.las"

        result = cli_runner.invoke(main, [
            'generate-sample',
            '--output', str(output_file),
            '--size', '20,20',
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_generate_sample_invalid_size(self, cli_runner, tmp_output_dir):
        """Test generate-sample with invalid size format."""
        output_file = tmp_output_dir / "sample.xyz"

        result = cli_runner.invoke(main, [
            'generate-sample',
            '--output', str(output_file),
            '--size', '20',  # Missing height
        ])

        assert result.exit_code != 0
        assert "width,height" in result.output


class TestCLIOptimize:
    """Test 'optimize' command."""

    @pytest.mark.requires_laspy
    @pytest.mark.slow
    def test_optimize_balance(self, cli_runner, demo_las_path):
        """Test optimization with balance constraint."""
        result = cli_runner.invoke(main, [
            'optimize', str(demo_las_path),
            '--polygon', '[[10,10],[30,10],[30,30],[10,30]]',
            '--constraint', 'balance',
            '--resolution', '2.0',
        ])

        assert result.exit_code == 0
        assert 'OPTIMIZATION RESULT' in result.output
        assert 'Optimal Elevation' in result.output


class TestCLIToDem:
    """Test 'to-dem' command."""

    @pytest.mark.requires_laspy
    @pytest.mark.requires_rasterio
    @pytest.mark.slow
    def test_to_dem(self, cli_runner, demo_las_path, tmp_output_dir):
        """Test conversion to DEM GeoTIFF."""
        output_file = tmp_output_dir / "terrain.tif"

        result = cli_runner.invoke(main, [
            'to-dem', str(demo_las_path),
            '--output', str(output_file),
            '--resolution', '2.0',
        ])

        assert result.exit_code == 0
        assert output_file.exists()


class TestCLIExport:
    """Test export options in analyze command."""

    @pytest.mark.requires_laspy
    @pytest.mark.slow
    def test_export_csv(self, cli_runner, demo_las_path, tmp_output_dir):
        """Test CSV export option."""
        csv_file = tmp_output_dir / "cells.csv"

        result = cli_runner.invoke(main, [
            'analyze', str(demo_las_path),
            '--elevation', '100.0',
            '--rect', '10,10,20,20',
            '--resolution', '2.0',
            '--export-csv', str(csv_file),
        ])

        assert result.exit_code == 0
        assert csv_file.exists()
        assert 'Cell details exported to' in result.output

        # Verify CSV has content
        content = csv_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) > 1  # Header + data

    @pytest.mark.requires_laspy
    @pytest.mark.slow
    def test_export_geojson(self, cli_runner, demo_las_path, tmp_output_dir):
        """Test GeoJSON export option."""
        geojson_file = tmp_output_dir / "zones.geojson"

        result = cli_runner.invoke(main, [
            'analyze', str(demo_las_path),
            '--elevation', '100.0',
            '--rect', '10,10,20,20',
            '--resolution', '2.0',
            '--export-geojson', str(geojson_file),
        ])

        assert result.exit_code == 0
        assert geojson_file.exists()
        assert 'Cut/fill zones exported to' in result.output

        # Verify GeoJSON structure
        with open(geojson_file) as f:
            data = json.load(f)
        assert data['type'] == 'FeatureCollection'
        assert 'features' in data
