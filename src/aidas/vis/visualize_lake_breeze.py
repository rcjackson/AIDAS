import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyart
import matplotlib.pyplot as plt
import numpy as np
import warnings

from ..io import RadarImage

def visualize_lake_breeze(radar_scan: RadarImage, bg_field='reflectivity', time=None, **kwargs):
    """
    This module will visualize the lake breeze using PyART's :py:meth:`pyart.graph.RadarMapDisplay`. The 
    default settings are geared towards Chicago with the proper latitude and longitude ranges.

    Keyword arguments are passed into :py:meth:`pyart.graph.RadarMapDisplay.plot_ppi_map`. See the 
    :py:meth:`pyart.graph.RadarMapDisplay.plot_ppi_map` documentation for more details on how to control
    the appearance of your plot.

    Parameters
    ----------
    radar_scan: :py:meth:`RadarImage`
        This contains the :py:meth:`RadarImage` object with the inferred lake breeze mask already inside.
    bg_field: str
        The variable name of the field you want to plot the lake breeze mask over.
    time: str
        The time, in format YYYY-MM-DDTHH:MM:SS, of the radar scan to plot if the input RadarImage
        contains batch processed radar images. AIDAS will look for the closest radar scan to this specified
        time when searching for scans.

    Returns
    -------
    fig, ax: handles
        The matplotlib figure and axis handle for the plot.
    """
    lat_lines = kwargs.pop('lat_lines', [41.2, 41.4, 41.6, 41.8, 42, 42.2, 42.4])
    lon_lines = kwargs.pop('lon_lines', [-88.4, -88.2, -88, -87.8, -87.6])
    vmin = kwargs.pop('vmin', -30)
    vmax = kwargs.pop('vmax', 60)
    sweep = kwargs.pop('sweep', 0)
    
    if isinstance(radar_scan.pyart_object, list) or isinstance(radar_scan.pyart_object, np.ndarray):
        my_times = np.array(radar_scan.times)
        if time is None:
            raise ValueError("Since the input radar_scan represents a batch of radar files, the UTC time in YYYY-MM-DDTHH:MM:SS format must be specified!")
        target_time = np.datetime64(time)
        my_ind = np.argmin(np.abs(my_times - target_time))
        pyart_obj = pyart.io.read(radar_scan.pyart_object[my_ind])
        lakebreeze_mask = radar_scan.lakebreeze_mask[my_ind].T
    else:
        pyart_obj = radar_scan.pyart_object
        lakebreeze_mask = radar_scan.lakebreeze_mask.squeeze().T
    disp = pyart.graph.RadarMapDisplay(pyart_obj)
    if 'ax' not in kwargs.keys():
        fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                subplot_kw=dict(projection=ccrs.PlateCarree()))
    else:
        fig = kwargs.pop('fig', plt.gcf())
        ax = kwargs.pop('ax')

    disp.plot_ppi_map(bg_field, sweep=sweep, min_lon=radar_scan.lon_range[0],
        ax=ax, max_lon=radar_scan.lon_range[1], 
        min_lat=radar_scan.lat_range[0], max_lat=radar_scan.lat_range[1],
        lat_lines=lat_lines, lon_lines=lon_lines, vmin=vmin, vmax=vmax, **kwargs)
    ax.coastlines()
    ax.add_feature(cfeature.STATES)
    ax.contour(radar_scan.grid_lon, radar_scan.grid_lat,
               lakebreeze_mask, levels=1)
    if isinstance(radar_scan.pyart_object, list):
        del pyart_obj
    return fig, ax

