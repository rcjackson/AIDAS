import pytest
import aidas
import torch
import numpy as np


class _MockRadarImage:
    """Minimal RadarImage stand-in for unit tests."""

    def __init__(self, mask, grid_lat, grid_lon, grid_x, grid_y):
        self.lakebreeze_mask = mask
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.grid_x = grid_x
        self.grid_y = grid_y

    def __getitem__(self, key):
        return self.lakebreeze_mask


def _make_ellipse_radar_image(N=256, semi_major=60, semi_minor=15):
    """
    Return a mock RadarImage containing a single horizontal ellipse
    (major axis east-west) centred on the grid.

    Inside azimuth_from_ellipse, mask.T is taken before extracting pixel
    coordinates, so mask[xi, yj] = 1 means the point at (grid_x[xi], grid_y[yj]).
    """
    grid_x = np.linspace(-100_000, 100_000, N)   # metres, easting
    grid_y = np.linspace(-100_000, 100_000, N)   # metres, northing
    grid_lon = np.linspace(-89.0, -88.0, N)
    grid_lat = np.linspace(41.0, 42.0, N)

    ci, cj = N // 2, N // 2  # centre at grid mid-point
    XI, YJ = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    mask = (
        (XI - ci) ** 2 / semi_major ** 2 + (YJ - cj) ** 2 / semi_minor ** 2 <= 1
    ).astype(float)

    return _MockRadarImage(mask, grid_lat, grid_lon, grid_x, grid_y), ci, cj, semi_major


def test_azimuth_from_ellipse_horizontal():
    """
    Horizontal E-W ellipse, instrument placed due south.

    The major axis runs east-west (azimuth 90°), so the perpendicular that
    points toward the ellipse centre from the south is due north (azimuth 0°).
    Also verifies that the left/right tips span the correct longitude range and
    that the reported centre matches the ellipse centroid.
    """
    rad_image, ci, cj, semi_major = _make_ellipse_radar_image()
    N = 256
    grid_x = rad_image.grid_x
    grid_y = rad_image.grid_y
    grid_lon = rad_image.grid_lon
    grid_lat = rad_image.grid_lat

    # Instrument due south of the ellipse centre (y ≈ −80 km, x = 0)
    inst_yi = np.argmin(np.abs(grid_y - (-80_000)))
    inst_xi = np.argmin(np.abs(grid_x - 0.0))
    instrument_lat = grid_lat[inst_yi]
    instrument_lon = grid_lon[inst_xi]

    azimuth, left_point, right_point, center_point = aidas.util.azimuth_from_ellipse(
        instrument_lon, instrument_lat, rad_image)

    # Azimuth should be ~0° (north): sin ≈ 0, cos > 0
    assert abs(np.sin(np.radians(azimuth))) < 0.05, (
        f"Expected azimuth ~0° (north), got {azimuth:.2f}°")
    assert np.cos(np.radians(azimuth)) > 0, (
        f"Azimuth should have a northward component, got {azimuth:.2f}°")

    # Left and right tips should be at the eastern and western ends of the ellipse.
    # The tips are ±semi_major pixels from the centre along axis 0 (→ grid_x/lon).
    expected_half_span_lon = semi_major / (N - 1) * (grid_lon[-1] - grid_lon[0])
    actual_half_span_lon = abs(left_point[1] - right_point[1]) / 2
    np.testing.assert_almost_equal(actual_half_span_lon, expected_half_span_lon, decimal=1)

    # Centre should be at the grid mid-point
    np.testing.assert_almost_equal(center_point[0], grid_lat[cj], decimal=1)
    np.testing.assert_almost_equal(center_point[1], grid_lon[ci], decimal=1)


def test_azimuth_from_ellipse_empty_mask():
    """An all-zero mask (no lake breeze) must raise ValueError."""
    N = 64
    rad_image = _MockRadarImage(
        mask=np.zeros((N, N)),
        grid_lat=np.linspace(41.0, 42.0, N),
        grid_lon=np.linspace(-89.0, -88.0, N),
        grid_x=np.linspace(-100_000, 100_000, N),
        grid_y=np.linspace(-100_000, 100_000, N),
    )
    with pytest.raises(ValueError):
        aidas.util.azimuth_from_ellipse(-88.5, 41.5, rad_image)


def test_instrument_steering():
    torch.manual_seed(42)
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-04-24T20:03:23')
    rad_scan = aidas.model.infer_lake_breeze(
        rad_scan, model_name='lakebreeze_model_fcn_resnet50_no_augmentation')
    angle, lat, lon, dist = aidas.util.azimuth_point(-87.99577278662817, 41.70101404798476, rad_scan)
    np.testing.assert_almost_equal(angle, 224.61, decimal=0)
    np.testing.assert_almost_equal(lat, 41.68, decimal=2)
    np.testing.assert_almost_equal(lon, -88.01, decimal=2)
    np.testing.assert_almost_equal(dist, 0, decimal=2)