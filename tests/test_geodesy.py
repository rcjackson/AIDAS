import numpy as np
import pytest

from aidas.util import aeqd_to_lonlat

# KLOT, the radar the rest of the suite uses.
LON_0, LAT_0 = -88.084444, 41.604444


def test_centre_maps_to_itself():
    lon, lat = aeqd_to_lonlat(0.0, 0.0, LON_0, LAT_0)
    assert lon == pytest.approx(LON_0, abs=1e-12)
    assert lat == pytest.approx(LAT_0, abs=1e-12)


def test_cardinal_directions():
    # x is east, y is north, so each cardinal offset moves one coordinate only.
    lon_n, lat_n = aeqd_to_lonlat(0.0, 50e3, LON_0, LAT_0)
    lon_s, lat_s = aeqd_to_lonlat(0.0, -50e3, LON_0, LAT_0)
    lon_e, lat_e = aeqd_to_lonlat(50e3, 0.0, LON_0, LAT_0)
    lon_w, lat_w = aeqd_to_lonlat(-50e3, 0.0, LON_0, LAT_0)

    # Due north and south hold longitude exactly.
    assert lon_n == pytest.approx(LON_0, abs=1e-9)
    assert lon_s == pytest.approx(LON_0, abs=1e-9)
    # They span nearly the same latitude range, but not exactly: the meridian
    # radius of curvature grows with latitude, so the same 50 km covers
    # marginally fewer degrees going north than going south.
    assert lat_n > LAT_0 > lat_s
    assert (lat_n - LAT_0) == pytest.approx(LAT_0 - lat_s, abs=1e-3)

    # Due east and west move longitude in opposite directions by equal amounts.
    assert lon_e > LON_0 and lon_w < LON_0
    assert (lon_e - LON_0) == pytest.approx(LON_0 - lon_w, abs=1e-9)

    # A geodesic launched due east curves towards the equator, so latitude does
    # not stay put -- but it drifts the same way on both sides, and only
    # slightly. Asserting exact constancy here would be wrong on an ellipsoid.
    assert lat_e == pytest.approx(lat_w, abs=1e-12)
    assert lat_e < LAT_0
    assert abs(lat_e - LAT_0) < 0.01


def test_shape_is_preserved():
    x = np.linspace(-100e3, 100e3, 12).reshape(3, 4)
    y = np.linspace(-100e3, 100e3, 12).reshape(3, 4)
    lon, lat = aeqd_to_lonlat(x, y, LON_0, LAT_0)
    assert lon.shape == (3, 4)
    assert lat.shape == (3, 4)


def test_matches_pyproj():
    """
    The model input depends on this geolocation, so it must agree with the
    reference implementation. AIDAS computes it here rather than calling PROJ so
    that a PROJ release cannot silently move every radar gate, which is what
    broke the triggering tests when PROJ 9.8 changed its equidistant
    cylindrical projection.
    """
    pyproj = pytest.importorskip('pyproj')

    angles = np.linspace(0, 2 * np.pi, 181)[:-1]
    ranges = np.array([1e3, 25e3, 75e3, 150e3, 230e3, 300e3])
    x = np.outer(ranges, np.sin(angles)).ravel()
    y = np.outer(ranges, np.cos(angles)).ravel()

    lon, lat = aeqd_to_lonlat(x, y, LON_0, LAT_0)

    transformer = pyproj.Transformer.from_crs(
        f"+proj=aeqd +ellps=WGS84 +lon_0={LON_0} +lat_0={LAT_0}",
        "EPSG:4326", always_xy=True)
    lon_ref, lat_ref = transformer.transform(x, y)

    _, _, separation = pyproj.Geod(ellps='WGS84').inv(lon_ref, lat_ref, lon, lat)
    # Agreement is a few microns; 1 mm leaves room for platform differences and
    # is still ~six orders of magnitude below one image pixel (~470 m).
    assert np.abs(separation).max() < 1e-3
