"""Utilities for pointing an instrument at a detected lake breeze front."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import center_of_mass, label

if TYPE_CHECKING:
    # Imported for type annotations only. A runtime import would be circular:
    # aidas.io reaches into aidas.util for the gate geolocation helper.
    from aidas.io import RadarImage

def azimuth_point(instrument_lon, instrument_lat, 
                  radar_image: RadarImage, index=None,
                  area_threshold=20):
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

    Returns
    -------
    deg_angle: float
       Azimuth angle in degrees from the radar instrument to the center of the largest lake breeze region.
    lat_center: float
       Latitude of the center of the largest lake breeze region.
    lon_center: float
       Longitude of the center of the largest lake breeze region.
    dist: float
       Distance in the radar image's grid units from the instrument to the
       nearest point in the lake breeze region.
    """
    # Convert lat/lon to radians
    if index is None:
        mask = radar_image[0]
    else:
        mask = radar_image[index]
    
    lats = radar_image.grid_lat
    lons = radar_image.grid_lon
    lat_index = np.argmin(np.abs(lats - instrument_lat))
    lon_index = np.argmin(np.abs(lons - instrument_lon))
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

    center = center_of_mass(mask)
    num_y = len(radar_image.grid_y)
    num_x = len(radar_image.grid_x)
    center_x = radar_image.grid_x[int(center[1])]
    center_y = radar_image.grid_y[int(center[0])]
    logging.info(f"Center of mass: {center}, Center lat/lon: {center_y}, {center_x}")
    instrument_x = radar_image.grid_x[lon_index]
    instrument_y = radar_image.grid_y[lat_index]
    logging.info(f"Instrument lat/lon: {instrument_y}, {instrument_x}")
    angle = np.arctan2((center_x - instrument_x), (center_y - instrument_y))
    deg_angle = np.rad2deg(angle)
    deg_angle = (deg_angle + 360) % 360

    # Get the distance from the instrument to the nearest point in the lake breeze region
    x, y = np.meshgrid(radar_image.grid_x, radar_image.grid_y)
    dist = np.sqrt((x - instrument_x)**2 + (y - instrument_y)**2)
    dist = dist[mask == 1]
    dist = np.min(dist)
    return deg_angle, lats[int(center[0])], lons[int(center[1])], dist


def azimuth_from_ellipse(instrument_lon, instrument_lat,
                          radar_image: RadarImage, index=None,
                          area_threshold=20):
    """
    Fit an ellipse to the lake breeze mask via PCA, find the endpoints of the
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
    if index is None:
        mask = radar_image[0]
    else:
        mask = radar_image[index]

    lats = radar_image.grid_lat
    lons = radar_image.grid_lon

    lat_index = np.argmin(np.abs(lats - instrument_lat))
    lon_index = np.argmin(np.abs(lons - instrument_lon))
    instrument_x = radar_image.grid_x[lon_index]
    instrument_y = radar_image.grid_y[lat_index]

    # Filter small regions (mirrors azimuth_point behaviour)
    labels, num_features = label(mask)
    mask = mask.T
    for i in range(num_features):
        area = mask[labels == i].sum()
        if area < area_threshold:
            mask[labels == i] = 0

    rows, cols = np.where(mask == 1)
    if len(rows) == 0:
        raise ValueError("No lake breeze pixels remaining after area filtering.")

    # Physical (metre) coordinates of each lake-breeze pixel
    xs = radar_image.grid_x[cols]
    ys = radar_image.grid_y[rows]

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
        xi = np.argmin(np.abs(radar_image.grid_x - x))
        yi = np.argmin(np.abs(radar_image.grid_y - y))
        return float(lats[yi]), float(lons[xi])

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
