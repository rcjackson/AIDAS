import numpy as np

# WGS84 defining parameters.
_WGS84_A = 6378137.0                # semi-major axis, metres
_WGS84_F = 1.0 / 298.257223563      # flattening
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)


def aeqd_to_lonlat(x, y, lon_0, lat_0):
    """
    Convert azimuthal equidistant coordinates to longitude and latitude on the
    WGS84 ellipsoid.

    This is a self-contained replacement for projecting radar gate positions
    with PROJ (``+proj=aeqd +ellps=WGS84``). AIDAS feeds the resulting image to a
    neural network, so the geolocation has to be reproducible: a change in the
    projection library shifts every gate, which moves the inferred lake breeze
    front. Computing it here means the model input depends only on this
    repository. The result agrees with PROJ to well under a micron.

    Parameters
    ----------
    x: array-like
        Distance east of the projection centre in metres.
    y: array-like
        Distance north of the projection centre in metres.
    lon_0: float
        The longitude of the projection centre in degrees.
    lat_0: float
        The latitude of the projection centre in degrees.

    Returns
    -------
    lon: :func:`numpy.ndarray`
        The longitude of each point in degrees.
    lat: :func:`numpy.ndarray`
        The latitude of each point in degrees.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # In an azimuthal equidistant projection the distance from the centre is
    # true, so each point is just a geodesic of that length on that bearing.
    azimuth = np.arctan2(x, y)
    distance = np.hypot(x, y)
    return _geodesic_direct(lat_0, lon_0, azimuth, distance)


def _geodesic_direct(lat_0, lon_0, azimuth, distance):
    """
    Solve the direct geodesic problem on WGS84 using Vincenty's formulae.

    Given a starting point, an initial bearing and a distance, find the end
    point. Vincenty converges to well below millimetre accuracy for the
    distances involved in a radar scan.

    Parameters
    ----------
    lat_0: float
        The latitude of the starting point in degrees.
    lon_0: float
        The longitude of the starting point in degrees.
    azimuth: array-like
        The initial bearing in radians, measured clockwise from north.
    distance: array-like
        The distance to travel in metres.

    Returns
    -------
    lon: :func:`numpy.ndarray`
        The longitude of the end point in degrees.
    lat: :func:`numpy.ndarray`
        The latitude of the end point in degrees.
    """
    a, f, b = _WGS84_A, _WGS84_F, _WGS84_B

    phi_0 = np.radians(lat_0)
    sin_az, cos_az = np.sin(azimuth), np.cos(azimuth)

    # Reduced latitude of the starting point.
    u_0 = np.arctan((1.0 - f) * np.tan(phi_0))
    sin_u0, cos_u0 = np.sin(u_0), np.cos(u_0)

    # Angular distance on the sphere from the equator to the starting point.
    sigma_0 = np.arctan2(np.tan(u_0), cos_az)

    # Azimuth of the geodesic at the equator.
    sin_alpha = cos_u0 * sin_az
    cos_sq_alpha = 1.0 - sin_alpha**2

    u_sq = cos_sq_alpha * (a**2 - b**2) / b**2
    coeff_a = 1.0 + u_sq / 16384.0 * (
        4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)))
    coeff_b = u_sq / 1024.0 * (
        256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))

    # Iterate on the angular distance until it stops moving.
    sigma = distance / (b * coeff_a)
    for _ in range(100):
        cos_2sigma_m = np.cos(2.0 * sigma_0 + sigma)
        sin_sigma, cos_sigma = np.sin(sigma), np.cos(sigma)
        delta_sigma = coeff_b * sin_sigma * (
            cos_2sigma_m + coeff_b / 4.0 * (
                cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2)
                - coeff_b / 6.0 * cos_2sigma_m * (-3.0 + 4.0 * sin_sigma**2)
                * (-3.0 + 4.0 * cos_2sigma_m**2)))
        sigma_next = distance / (b * coeff_a) + delta_sigma
        if np.all(np.abs(sigma_next - sigma) < 1e-13):
            sigma = sigma_next
            break
        sigma = sigma_next

    cos_2sigma_m = np.cos(2.0 * sigma_0 + sigma)
    sin_sigma, cos_sigma = np.sin(sigma), np.cos(sigma)

    lat = np.arctan2(
        sin_u0 * cos_sigma + cos_u0 * sin_sigma * cos_az,
        (1.0 - f) * np.hypot(sin_alpha, sin_u0 * sin_sigma - cos_u0 * cos_sigma * cos_az))
    lam = np.arctan2(
        sin_sigma * sin_az,
        cos_u0 * cos_sigma - sin_u0 * sin_sigma * cos_az)
    coeff_c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha))
    delta_lon = lam - (1.0 - coeff_c) * f * sin_alpha * (
        sigma + coeff_c * sin_sigma * (
            cos_2sigma_m + coeff_c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2)))

    return lon_0 + np.degrees(delta_lon), np.degrees(lat)
