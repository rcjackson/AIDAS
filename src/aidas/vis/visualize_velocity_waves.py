import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyart

from ..io import RadarImage


def visualize_velocity_waves(radar_scan: RadarImage, bg_field='velocity', **kwargs):
    """
    This module will visualize the radial velocity wave mask using PyART's
    :py:meth:`pyart.graph.RadarMapDisplay`.

    The background defaults to the radial velocity field the detection was made on,
    so the alternating bands the mask is drawn from are visible underneath it. The
    map is framed on the wave grid rather than on a fixed domain, because the grid
    reaches as far as the Doppler cut has valid data.

    Keyword arguments are passed into :py:meth:`pyart.graph.RadarMapDisplay.plot_ppi_map`.
    See the :py:meth:`pyart.graph.RadarMapDisplay.plot_ppi_map` documentation for more
    details on how to control the appearance of your plot.

    Parameters
    ----------
    radar_scan: :py:meth:`RadarImage`
        This contains the :py:meth:`RadarImage` object with the velocity wave mask
        already inside, as made by :func:`aidas.model.detect_velocity_waves`.
    bg_field: str
        The variable name of the field you want to plot the wave mask over.

    Returns
    -------
    fig, ax: handles
        The matplotlib figure and axis handle for the plot.
    """
    if radar_scan.velocity_wave_mask is None:
        raise ValueError(
            "This RadarImage has no velocity wave mask. Run aidas.model.detect_velocity_waves "
            "on it first.")

    vmin = kwargs.pop('vmin', -30)
    vmax = kwargs.pop('vmax', 30)
    cmap = kwargs.pop('cmap', 'balance')
    colors = kwargs.pop('colors', 'k')
    linewidths = kwargs.pop('linewidths', 0.6)

    pyart_obj = radar_scan.pyart_object
    if isinstance(pyart_obj, (list, np.ndarray)):
        raise ValueError(
            "Wave detection works on a single pair of volumes, so the RadarImage should hold "
            "one radar object rather than a batch.")
    if isinstance(pyart_obj, str):
        pyart_obj = pyart.io.read(pyart_obj)

    lon = radar_scan.wave_grid_lon
    lat = radar_scan.wave_grid_lat
    # Plot the sweep the mask was actually made from. On a NEXRAD split cut that is
    # the Doppler cut rather than the first sweep at that elevation, so guessing it
    # from the sweep order would put the mask over the wrong picture.
    sweep = kwargs.pop('sweep', radar_scan.wave_sweep if radar_scan.wave_sweep is not None else 0)

    disp = pyart.graph.RadarMapDisplay(pyart_obj)
    if 'ax' not in kwargs.keys():
        fig, ax = plt.subplots(1, 1, figsize=(6, 6),
                subplot_kw=dict(projection=ccrs.PlateCarree()))
    else:
        fig = kwargs.pop('fig', plt.gcf())
        ax = kwargs.pop('ax')

    disp.plot_ppi_map(bg_field, sweep=sweep, ax=ax,
        min_lon=float(lon.min()), max_lon=float(lon.max()),
        min_lat=float(lat.min()), max_lat=float(lat.max()),
        vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    ax.coastlines()
    ax.add_feature(cfeature.STATES)
    ax.contour(lon, lat, radar_scan.velocity_wave_mask, levels=[0.5],
               colors=colors, linewidths=linewidths)
    return fig, ax
