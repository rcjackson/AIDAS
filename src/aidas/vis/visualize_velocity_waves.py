import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyart
from matplotlib.colors import ListedColormap

from ..io import RadarImage


def _add_map_features(ax, lon, lat):
    """
    Frame an axes on the wave grid and draw the coastline and state borders.

    Parameters
    ----------
    ax: :py:meth:`cartopy.mpl.geoaxes.GeoAxes`
        The axes to draw on.
    lon, lat: ndarray
        The longitude and latitude of the wave grid, used to set the extent.
    """
    ax.set_extent([float(lon.min()), float(lon.max()),
                   float(lat.min()), float(lat.max())], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES)


def plot_velocity_wave_mask(radar_scan: RadarImage, ax=None, color='#1b3a6b',
                            title=None, gridlines=True):
    """
    Draw the radial velocity wave mask on its own map.

    The mask is binary, so it is drawn as one flat colour on white rather than with a
    colour scale. On its own like this the spacing and orientation of the bands are
    easy to read, which they are not when the mask is a contour over a velocity field.

    Parameters
    ----------
    radar_scan: :py:meth:`RadarImage`
        The :py:meth:`RadarImage` carrying the mask, as made by
        :func:`aidas.model.detect_velocity_waves`.
    ax: :py:meth:`cartopy.mpl.geoaxes.GeoAxes`, optional
        The axes to draw into, which must be on a :py:meth:`cartopy.crs.PlateCarree`
        projection. If None, a new figure and axes are made.
    color: str
        The colour to draw detections in.
    title: str, optional
        The title for the panel. If None, the two sweep times and the interval
        between them are used.
    gridlines: bool
        Whether to draw labelled latitude and longitude gridlines.

    Returns
    -------
    fig, ax: handles
        The matplotlib figure and axis handle for the plot.
    """
    if radar_scan.velocity_wave_mask is None:
        raise ValueError(
            "This RadarImage has no velocity wave mask. Run aidas.model.detect_velocity_waves "
            "on it first.")

    lon = radar_scan.wave_grid_lon
    lat = radar_scan.wave_grid_lat

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5.5),
                subplot_kw=dict(projection=ccrs.PlateCarree()))
    else:
        fig = ax.get_figure()

    # White where nothing was detected so the map shows through, one flat colour
    # where it was. The mask is binary, so a colour scale would say nothing.
    ax.pcolormesh(lon, lat, radar_scan.velocity_wave_mask,
                  cmap=ListedColormap(['white', color]), vmin=0, vmax=1, shading='auto')
    _add_map_features(ax, lon, lat)
    if gridlines:
        lines = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray')
        lines.top_labels = False
        lines.right_labels = False

    if title is None:
        times = radar_scan.wave_scan_times
        if times is None:
            title = "Wave mask"
        else:
            gap = int((times[1] - times[0]) / np.timedelta64(1, 's'))
            title = f"Wave mask\n{times[0]} to {times[1]} ({gap} s apart)"
    ax.set_title(title)
    return fig, ax


def visualize_velocity_waves(radar_scan: RadarImage, bg_field='velocity', axes=None,
                             overlay=False, **kwargs):
    """
    This module will visualize the radial velocity wave mask alongside the velocity
    field it was derived from.

    Two panels are drawn. The left one is the radial velocity of the sweep the
    detection used, after the quality control and the dealiasing: that is the field
    the mask was computed from, not the raw folded velocity in the original volume,
    so what is plotted and what was detected agree. The right one is the binary mask
    on its own, which is easier to read for band spacing and orientation than a
    contour drawn over a colourful velocity field.

    The map is framed on the wave grid rather than on a fixed domain, because the
    grid reaches as far as the sweep has valid velocity.

    Keyword arguments are passed into :py:meth:`pyart.graph.RadarMapDisplay.plot_ppi_map`.
    See the :py:meth:`pyart.graph.RadarMapDisplay.plot_ppi_map` documentation for more
    details on how to control the appearance of the velocity panel.

    Parameters
    ----------
    radar_scan: :py:meth:`RadarImage`
        This contains the :py:meth:`RadarImage` object with the velocity wave mask
        already inside, as made by :func:`aidas.model.detect_velocity_waves`.
    bg_field: str
        The variable name of the field to draw in the left panel. When it is a
        velocity field, the colour scale defaults to a symmetric range wide enough
        for the dealiased values rather than to the Nyquist velocity.
    axes: 2-sequence of :py:meth:`cartopy.mpl.geoaxes.GeoAxes`, optional
        The axes to draw the velocity and the mask into. Both must be on a
        :py:meth:`cartopy.crs.PlateCarree` projection. If None, a new figure is made.
    overlay: bool
        Set to True to also outline the mask on the velocity panel, which is useful
        for checking that the detections sit on the bands they came from.

    Returns
    -------
    fig, axes: handles
        The matplotlib figure and the two axis handles, velocity first.
    """
    if radar_scan.velocity_wave_mask is None:
        raise ValueError(
            "This RadarImage has no velocity wave mask. Run aidas.model.detect_velocity_waves "
            "on it first.")

    mask_color = kwargs.pop('mask_color', '#1b3a6b')

    # Prefer the sweep the detection kept, which holds the dealiased velocity. Fall
    # back to the full volume for a RadarImage made before that was stored.
    pyart_obj = radar_scan.wave_sweep_radar
    sweep = 0
    if pyart_obj is None:
        pyart_obj = radar_scan.pyart_object
        sweep = radar_scan.wave_sweep if radar_scan.wave_sweep is not None else 0
        if isinstance(pyart_obj, (list, np.ndarray)):
            raise ValueError(
                "Wave detection works on a single pair of volumes, so the RadarImage should "
                "hold one radar object rather than a batch.")
        if isinstance(pyart_obj, str):
            pyart_obj = pyart.io.read(pyart_obj)
    sweep = kwargs.pop('sweep', sweep)

    lon = radar_scan.wave_grid_lon
    lat = radar_scan.wave_grid_lat
    times = radar_scan.wave_scan_times

    # Py-ART's defaults here are a long title and a longer colourbar label, which
    # collide with the second panel.
    title = kwargs.pop('title', "Radial velocity used for detection"
                       + (f"\n{times[1]}" if times is not None else ""))
    colorbar_label = kwargs.pop('colorbar_label', 'Radial velocity (m s$^{-1}$)')

    if 'velocity' in bg_field:
        # Dealiasing is the point of the left panel, so the colour scale has to
        # reach past the Nyquist velocity. Unfolded gates run to several times it,
        # and a fixed scale would fold them back visually after Py-ART had just
        # taken the trouble to unfold them.
        values = np.ma.compressed(np.ma.abs(pyart_obj.fields[bg_field]['data']))
        limit = float(np.ceil(np.percentile(values, 99.5) / 5.0) * 5.0) if values.size else 30.0
        kwargs.setdefault('vmin', -max(limit, 10.0))
        kwargs.setdefault('vmax', max(limit, 10.0))
        kwargs.setdefault('cmap', 'balance')

    if axes is None:
        # The colourbar Py-ART adds sits between the panels, so they need room
        # between them or its label lands on the mask panel's latitude labels.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                subplot_kw=dict(projection=ccrs.PlateCarree()),
                gridspec_kw={'wspace': 0.45})
    else:
        fig = kwargs.pop('fig', plt.gcf())
    velocity_ax, mask_ax = axes[0], axes[1]

    pyart.graph.RadarMapDisplay(pyart_obj).plot_ppi_map(
        bg_field, sweep=sweep, ax=velocity_ax,
        min_lon=float(lon.min()), max_lon=float(lon.max()),
        min_lat=float(lat.min()), max_lat=float(lat.max()),
        title=title, colorbar_label=colorbar_label, **kwargs)
    _add_map_features(velocity_ax, lon, lat)
    if overlay:
        velocity_ax.contour(lon, lat, radar_scan.velocity_wave_mask, levels=[0.5],
                            colors='k', linewidths=0.6)

    plot_velocity_wave_mask(radar_scan, ax=mask_ax, color=mask_color)
    return fig, axes
