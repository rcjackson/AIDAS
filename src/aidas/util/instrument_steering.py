"""Utilities for pointing an instrument at a detected feature."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import center_of_mass, label

if TYPE_CHECKING:
    # Imported for type annotations only. A runtime import would be circular:
    # aidas.io reaches into aidas.util for the gate geolocation helper.
    from aidas.io import RadarImage


def _mask_and_grid(radar_image: RadarImage, mask, index):
    """
    Pick out the detection mask asked for, along with the grid it sits on.

    AIDAS has two kinds of mask and they are stored differently. The lake breeze
    mask is on the 256 by 256 latitude/longitude domain grid and is held
    transposed; the velocity wave mask is on a Cartesian grid centred on the radar,
    is held as ``[y, x]``, and carries two dimensional latitude and longitude
    because a radar-centred grid is not separable in latitude and longitude. This
    hands both back in the one convention the pointing maths uses, so that the
    callers differ only in which mask they ask for.

    Parameters
    ----------
    radar_image: RadarImage
        The RadarImage holding the mask.
    mask: str
        Which mask to steer from: 'lakebreeze' or 'velocity_wave'.
    index: int or None
        The time index, for a RadarImage holding several lake breeze masks.

    Returns
    -------
    mask: ndarray
        The mask indexed ``[x, y]``, which is what the callers transpose.
    grid_x, grid_y: ndarray
        The east and north coordinate of each grid column and row in metres.
    to_latlon: callable
        Maps a ``(row, column)`` index of the transposed mask to (lat, lon).
    instrument_index: callable
        Maps an instrument (lat, lon) to the nearest ``(row, column)``.
    """
    if mask == 'lakebreeze':
        lats = radar_image.grid_lat
        lons = radar_image.grid_lon
        return (radar_image[index], radar_image.grid_x, radar_image.grid_y,
                lambda row, col: (float(lats[row]), float(lons[col])),
                lambda lat, lon: (int(np.argmin(np.abs(lats - lat))),
                                  int(np.argmin(np.abs(lons - lon)))))

    if mask == 'velocity_wave':
        if radar_image.velocity_wave_mask is None:
            raise ValueError(
                "This RadarImage has no velocity wave mask. Run "
                "aidas.model.detect_velocity_waves on it first.")
        lat = radar_image.wave_grid_lat
        lon = radar_image.wave_grid_lon

        def _nearest_cell(instrument_lat, instrument_lon):
            # Scale the longitude difference by the cosine of the latitude so that
            # a degree of longitude counts for what it is worth on the ground.
            offset = ((lat - instrument_lat) ** 2
                      + ((lon - instrument_lon) * np.cos(np.radians(instrument_lat))) ** 2)
            return np.unravel_index(int(np.argmin(offset)), offset.shape)

        return (radar_image.velocity_wave_mask.T, radar_image.wave_grid_x,
                radar_image.wave_grid_y,
                lambda row, col: (float(lat[row, col]), float(lon[row, col])),
                _nearest_cell)

    raise ValueError(f"'{mask}' is not a mask AIDAS can steer from.")


def azimuth_point(instrument_lon, instrument_lat, 
                  radar_image: RadarImage, index=None,
                  area_threshold=20, mask='lakebreeze'):
    """
    Calculate the azimuth angle from the radar instrument to each pixel in the radar image.

    Parameters
    ----------
    instrument_lat: float
        Latitude of the radar instrument in degrees.
    instrument_lon: float
        Longitude of the radar instrument in degrees.
    radar_image: RadarImage
        The RadarImage object containing the radar data and metadata.
    index: int, optional
        If the radar image contains multiple time frames, specify the index of the frame to use. If None, use the first frame.
    area_threshold: int
        The minimum continuous area in pixels for lake breeze segments. This helps
        remove false positive speckles that are identified by the model.
    mask: str
        Which detection mask to point at: 'lakebreeze' for the inferred lake breeze
        front, or 'velocity_wave' for the radial velocity wave mask from
        :func:`aidas.model.detect_velocity_waves`.

    Returns
    -------
    deg_angle: float
       Azimuth angle in degrees from the radar instrument to the center of the largest detected region.
    lat_center: float
       Latitude of the center of the largest detected region.
    lon_center: float
       Longitude of the center of the largest detected region.
    dist: float
       Distance in the radar image's grid units from the instrument to the
       nearest point in the detected region.
    """
    # Convert lat/lon to radians
    mask_name = mask
    mask, grid_x, grid_y, to_latlon, instrument_index = _mask_and_grid(
        radar_image, mask_name, 0 if index is None else index)

    lat_index, lon_index = instrument_index(instrument_lat, instrument_lon)
    # The wave mask arrives already filtered, by area in square kilometres rather
    # than in pixels, which is the scale selection the detection method is built
    # around. Filtering it again here would only undo that.
    if mask_name == 'lakebreeze':
        labels, num_features = label(mask)
        area_threshold = 20
        largest_area = -99999
        mask = mask.T
        for i in range(num_features):
            area = mask[labels == i].sum()
            if area > largest_area:
                largest_area = area
            if area < area_threshold:
                mask[labels == i] = 0
    else:
        mask = mask.T

    center = center_of_mass(mask)
    center_x = grid_x[int(center[1])]
    center_y = grid_y[int(center[0])]
    logging.info(f"Center of mass: {center}, Center lat/lon: {center_y}, {center_x}")
    instrument_x = grid_x[lon_index]
    instrument_y = grid_y[lat_index]
    logging.info(f"Instrument lat/lon: {instrument_y}, {instrument_x}")
    angle = np.arctan2((center_x - instrument_x), (center_y - instrument_y))
    deg_angle = np.rad2deg(angle)
    deg_angle = (deg_angle + 360) % 360

    # Get the distance from the instrument to the nearest point in the detected region
    x, y = np.meshgrid(grid_x, grid_y)
    dist = np.sqrt((x - instrument_x)**2 + (y - instrument_y)**2)
    dist = dist[mask == 1]
    dist = np.min(dist)
    center_lat, center_lon = to_latlon(int(center[0]), int(center[1]))
    return deg_angle, center_lat, center_lon, dist


def azimuth_from_ellipse(instrument_lon, instrument_lat,
                          radar_image: RadarImage, index=None,
                          area_threshold=20, mask='lakebreeze'):
    """
    Fit an ellipse to the detection mask via PCA, find the endpoints of the
    major axis (leftmost and rightmost mask pixels projected onto that axis),
    and return the azimuth from the instrument perpendicular to the major axis.

    Parameters
    ----------
    instrument_lon : float
        Longitude of the instrument in degrees.
    instrument_lat : float
        Latitude of the instrument in degrees.
    radar_image : RadarImage
        The RadarImage object containing the radar data and metadata.
    index : int, optional
        Time index of the mask to use. If None, uses the first frame.
    area_threshold : int
        Minimum contiguous area in pixels to retain (removes small speckles).
    mask : str
        Which detection mask to fit: 'lakebreeze' for the inferred lake breeze
        front, or 'velocity_wave' for the radial velocity wave mask from
        :func:`aidas.model.detect_velocity_waves`. A wave train is a far better fit
        to an ellipse than a single front is, since its bands are parallel.

    Returns
    -------
    azimuth : float
        Azimuth angle in degrees from the instrument, pointing perpendicular
        to the major axis of the fitted ellipse and toward the ellipse center.
    left_point : tuple of (float, float)
        (lat, lon) of the endpoint of the major axis with the smallest
        projection value (i.e. the "leftmost" tip of the ellipse).
    right_point : tuple of (float, float)
        (lat, lon) of the endpoint of the major axis with the largest
        projection value (i.e. the "rightmost" tip of the ellipse).
    center_point : tuple of (float, float)
        (lat, lon) of the centroid of the lake breeze mask.
    """
    mask_name = mask
    mask, grid_x, grid_y, to_latlon, instrument_index = _mask_and_grid(
        radar_image, mask_name, 0 if index is None else index)

    lat_index, lon_index = instrument_index(instrument_lat, instrument_lon)
    instrument_x = grid_x[lon_index]
    instrument_y = grid_y[lat_index]

    # Filter small regions (mirrors azimuth_point behaviour). The wave mask has
    # already been filtered by area in square kilometres, so it is left alone.
    if mask_name == 'lakebreeze':
        labels, num_features = label(mask)
        mask = mask.T
        for i in range(num_features):
            area = mask[labels == i].sum()
            if area < area_threshold:
                mask[labels == i] = 0
    else:
        mask = mask.T

    rows, cols = np.where(mask == 1)
    if len(rows) == 0:
        raise ValueError("No detected pixels remaining after area filtering.")

    # Physical (metre) coordinates of each detected pixel
    xs = grid_x[cols]
    ys = grid_y[rows]

    # Centroid
    cx = np.mean(xs)
    cy = np.mean(ys)

    # PCA: covariance matrix → major axis = eigenvector of largest eigenvalue
    dx = xs - cx
    dy = ys - cy
    cov = np.array([[np.mean(dx ** 2), np.mean(dx * dy)],
                    [np.mean(dx * dy), np.mean(dy ** 2)]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]  # [east, north] unit vector

    # Project every pixel onto the major axis; extremes are the left/right tips
    projections = dx * major_axis[0] + dy * major_axis[1]
    left_x, left_y = xs[np.argmin(projections)], ys[np.argmin(projections)]
    right_x, right_y = xs[np.argmax(projections)], ys[np.argmax(projections)]

    def xy_to_latlon(x, y):
        xi = int(np.argmin(np.abs(grid_x - x)))
        yi = int(np.argmin(np.abs(grid_y - y)))
        return to_latlon(yi, xi)

    left_point = xy_to_latlon(left_x, left_y)
    right_point = xy_to_latlon(right_x, right_y)
    center_point = xy_to_latlon(cx, cy)

    # Azimuth of the major axis (degrees from north, clockwise)
    major_azimuth = np.rad2deg(np.arctan2(major_axis[0], major_axis[1]))

    # Two candidate perpendicular azimuths
    perp1 = (major_azimuth + 90) % 360
    perp2 = (major_azimuth - 90) % 360

    # Pick the perpendicular that points from the instrument toward the ellipse centre
    center_azimuth = np.rad2deg(
        np.arctan2(cx - instrument_x, cy - instrument_y)
    ) % 360

    def _angle_diff(a, b):
        d = abs((a - b) % 360)
        return min(d, 360 - d)

    azimuth = perp1 if _angle_diff(perp1, center_azimuth) <= _angle_diff(perp2, center_azimuth) else perp2

    logging.info(
        f"Ellipse major-axis azimuth: {major_azimuth:.1f}°, "
        f"perpendicular azimuth: {azimuth:.1f}°"
    )
    logging.info(f"Left tip: {left_point}, Right tip: {right_point}, Centre: {center_point}")

    return azimuth, left_point, right_point, center_point
