import boto3
import pyart
import numpy as np
import cmweather
import matplotlib.pyplot as plt
import torch
import os
import dask.bag as db
import tempfile

from glob import glob
from datetime import datetime, timedelta 
from botocore import UNSIGNED
from torchvision.io import decode_image
from torchvision import transforms
from botocore.config import Config

from ..util.geodesy import aeqd_to_lonlat

def _render_model_input(radar, lat_range, lon_range):
    """
    Rasterise the lowest reflectivity sweep into the 256x256 image the
    lake-breeze model expects.

    The gate geolocation is computed by :func:`aidas.util.aeqd_to_lonlat` and the
    figure is drawn on a plain Matplotlib axes rather than a Cartopy GeoAxes.
    Longitude and latitude are the axes coordinates directly, which is what a
    plate carree map is, so the picture is unchanged -- but nothing in the path
    consults a projection library. That matters because the image is model
    input: PROJ 9.8 altered its equidistant cylindrical projection and shifted
    every gate, which moved the inferred front by 17 km and broke the
    triggering tests. Keeping the geolocation here means a projection release
    cannot silently change what the network sees.

    Parameters
    ----------
    radar: :func:`pyart.core.Radar`
        The radar volume to render.
    lat_range: 2-tuple of floats
        The minimum and maximum latitude of the domain in degrees.
    lon_range: 2-tuple of floats
        The minimum and maximum longitude of the domain in degrees.

    Returns
    -------
    image: :func:`torch.Tensor`
        The normalized (1, 3, 256, 256) image ready for inference.
    """
    x, y, _ = radar.get_gate_x_y_z(0)
    lon_0 = float(radar.longitude['data'][0])
    lat_0 = float(radar.latitude['data'][0])
    lon, lat = aeqd_to_lonlat(x, y, lon_0, lat_0)
    reflectivity = radar.get_field(0, 'reflectivity')

    fig, ax = plt.subplots(1, 1, figsize=(2.56, 2.56), frameon=False)
    ax.pcolormesh(lon, lat, reflectivity, vmin=-20, vmax=60,
                  cmap='HomeyerRainbow', edgecolors='face')
    ax.set_xlim(lon_range[0], lon_range[1])
    ax.set_ylim(lat_range[0], lat_range[1])
    # A plate carree axes is square in degrees; match it so the aspect ratio of
    # the rendered image is the same as it has always been.
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    with tempfile.NamedTemporaryFile(mode='w+b') as temp_file:
        fig.savefig(temp_file, dpi=100)
        plt.close(fig)
        image = decode_image(temp_file.name)
    image = image[:3, :, :].float()
    transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return torch.stack([transform(image)])


class RadarImage(object):
    """
    This class contains the basic data for predicting the lake-breeze front location
    from radar. It includes parameters for storing the radar data for later analysis
    in Py-ART and for inference in the lake-breeze prediction model.

    Parameters
    ----------
    pyart_object: :py:meth:`pyart.core.Radar` or str
        The PyART radar object that stores the radar data. This could also be a link
        to the radar scan file (useful for batch processing to preserve memory).
    lat_range: 2-tuple
        The minimum and maximum latitude of the inference domain.
    lon_range: 2-tuple
        The minimum and maximum longitude of the inference domain.
    grid_lat: ndarray
        The latitude of each point in the inference domain.
    grid_lon: ndarray
        The longitude of each point in the inference domain.
    pytorch_image: :py:meth:`torch.Tensor`
        The tensor containing the preprocessed radar scan for inference.
    lakebreeze_mask: 256 x 256 ndarray
        The inferred lake breeze mask, where 1 = lakebreeze and 0 = not a lake breeze. 
    times: list of np.datetime64('s')
        The epoch time of the radar scans.
    """
    pyart_object = None
    lat_range = None
    lon_range = None
    grid_lat = None
    grid_lon = None
    pytorch_image = None
    lakebreeze_mask = None
    aggregated_mask = None
    times = None
    
    def __getitem__(self, key):
        """
        Allows for indexing into the RadarImage object to get the lake breeze mask
        for a specific time.
        
        Parameters
        ----------
        key: int or np.datetime64('s')
            The index or time to retrieve the lake breeze mask for.

        Returns
        -------
        mask: ndarray
            The lake breeze mask for the specified time.
        """
        if len(self.lakebreeze_mask.shape) == 2:
            return self.lakebreeze_mask
        if isinstance(key, int):
            return self.lakebreeze_mask[key]
        elif isinstance(key, np.datetime64):
            return self.aggregate(start_time=key, end_time=key)
        elif isinstance(key, str):
            key = np.datetime64(key)
            return self.aggregate(start_time=key, end_time=key)
        elif isinstance(key, slice):
            return self.lakebreeze_mask[key]
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            if all(isinstance(x, int) for x in key):
                return [self.lakebreeze_mask[x] for x in key]
            elif all(isinstance(x, (str, np.datetime64)) for x in key):
                masks = []
                for time in key:
                    if isinstance(time, str):
                        time = np.datetime64(time)
                    masks.append(self.aggregate(start_time=time, end_time=time))
                return masks
            else:
                raise TypeError("All keys in the list must be of the same type: int, str, or np.datetime64('s').")
        else:
            raise TypeError("Key must be an int, np.datetime64('s'), str, or slice.")
        
    def aggregate(self, start_time=None, end_time=None):
        """
        This function aggregates the lake breeze mask over a specified time period.
        If no time period is specified, it returns the sum of all of the masks.

        Parameters
        ----------
        start_time: str or np.datetime64('s')
            The start time for aggregation.
        end_time: str np.datetime64('s')
            The end time for aggregation.

        Returns
        -------
        aggregated_mask: RadarImage
            The aggregated lake breeze mask.
        """
        if start_time is None and end_time is None:
            self.aggregated_mask = np.sum(self.lakebreeze_mask, axis=0)
            return self.aggregated_mask

        mask = self.lakebreeze_mask.copy()
        if isinstance(start_time, str):
            start_time = np.datetime64(start_time)
        if isinstance(end_time, str):
            end_time = np.datetime64(end_time)
        if (start_time is None) ^ (end_time is None):
            raise ValueError("Both start_time and end_time must be specified for aggregation.")
        indices = np.where((self.times >= start_time) & (self.times <= end_time))[0]
        if len(indices) == 0:
            raise ValueError("No data available for the specified time range.")
        mask = mask[indices]
        self.aggregated_mask = np.sum(mask, axis=0)
        return self.aggregated_mask


def preprocess_radar_image(radar, rad_time=None, lat_range=(41.1280, 42.5680),
                           lon_range=(-88.7176, -87.2873),
                           bucket_name='unidata-nexrad-level2'):
    """
    This module will preprocess the NEXRAD radar data for inference into the lake-breeze
    prediction model of AIDAS.

    Parameters
    ----------
    radar: str or :py:meth:`pyart.core.radar` object
        The 4-letter code for the radar to obtain the scan from. For Chicago, use KLOT.
    rad_time: ISO-format datestring
        The date/time string in YYYY-MM-DDTHH:MM:SS format for the radar scan. If None, then AIDAS will
        get the latest scan. This is not used if radar is a :py:meth:`pyart.core.radar` object or string.
    lat_range: 2-tuple of floats
        The minimum and maximum latitude of the domain in degrees. Default is a centered
        domain around the KLOT Chicago area radar.
    lon_range: 2-tuple of floats
        The minimum and maximum longitude of the domain in degrees. Default is a centered
        domain around the KLOT Chicago area radar.
    bucket_name: str
        The NEXRAD S3 bucket to use. Default is 'unidata-nexrad-level2'.

    Returns
    -------
    image: :py:meth:`aidas.io.RadarImage`
        The :py:meth:`RadarImage` object containing the radar scan, pre-processed image,
        and grid.
    """
    if isinstance(radar, str):
        if rad_time is None:
            right_now = datetime.utcnow()
        else:
            right_now = datetime.strptime(rad_time, "%Y-%m-%dT%H:%M:%S")
        yesterday = right_now - timedelta(days=1)
        year = right_now.year
        month = right_now.month
        day = right_now.day

        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        bucket_name = 'unidata-nexrad-level2'
        radar = "KLOT"
        prefix = f'{year}/{month:02d}/{day:02d}/{radar}'
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        file_list = [x['Key'] for x in response['Contents']]
        
        # Find yesterday's scans
        prefix = f'{yesterday.year}/{yesterday.month:02d}/{yesterday.day:02d}/{radar}'
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        file_list = file_list + [x['Key'] for x in response['Contents']]
        time_list = []
        for filepath in file_list:
            name = filepath.split("/")[-1]
            if name[-3:] == "MDM":
                time_list.append(
                    datetime.strptime(name, f"{radar}%Y%m%d_%H%M%S_V06_MDM"))
            else:
                time_list.append(
                    datetime.strptime(name, f"{radar}%Y%m%d_%H%M%S_V06"))

        time_list = np.array(time_list)
        path = f"s3://{bucket_name}/" + file_list[np.argmin(np.abs(time_list - right_now))]
        cur_radar = pyart.io.read_nexrad_archive(path)
    elif isinstance(radar, pyart.core.Radar):
        cur_radar = radar
    elif isinstance(radar, str):
        cur_radar = pyart.io.read(str)
    else:
        raise ValueError("The radar input must be a string or a PyART radar object.")

    image = _render_model_input(cur_radar, lat_range, lon_range)
    lats = np.linspace(lat_range[1], lat_range[0], image.shape[3])
    lons = np.linspace(lon_range[0], lon_range[1], image.shape[2])

    rad_image = RadarImage()
    rad_image.pytorch_image = image
    rad_image.lat_range = lat_range
    rad_image.lon_range = lon_range
    rad_image.pyart_object = cur_radar
    rad_image.grid_lat = lats
    rad_image.grid_lon = lons
    center_lat = (rad_image.lat_range[0] + rad_image.lat_range[1]) / 2.
    center_lon = (rad_image.lon_range[0] + rad_image.lon_range[1]) / 2.
    rad_image.grid_x, rad_image.grid_y = _latlon_to_xy(
        rad_image.grid_lat, rad_image.grid_lon, center_lat, center_lon)
    rad_image.times = [np.datetime64(cur_radar.time["units"].split()[2])]
    
    return rad_image

def preprocess_radar_image_batch(file, lat_range=(41.1280, 42.5680),
                           lon_range=(-88.7176, -87.2873), parallel=False):
    """
    This module will preprocess the NEXRAD radar data for inference into the lake-breeze
    prediction model of AIDAS.

    Parameters
    ----------
    radar: str or list
        The 4-letter code for the radar to obtain the scan from. For Chicago, use KLOT.
    lat_range: 2-tuple of floats
        The minimum and maximum latitude of the domain in degrees. Default is a centered
        domain around the KLOT Chicago area radar.
    lon_range: 2-tuple of floats
        The minimum and maximum longitude of the domain in degrees. Default is a centered
        domain around the KLOT Chicago area radar.
    parallel: bool
        If true, enable parallel preprocessing for large radar datasets using Dask.

    Returns
    -------
    image: :py:meth:`aidas.io.RadarImage`
        The :py:meth:`RadarImage` object containing the radar scan, pre-processed image,
        and grid.
    """
    
    if isinstance(file, str):
        files = sorted(glob(file))
    else:
        files = file
    _pprocess = lambda x: _preprocess(x, lat_range, lon_range)

    if parallel:
        arr = db.from_sequence(files).map(_pprocess).compute()
        images = [x[0] for x in arr]
        times = [x[1] for x in arr]
    else:
        arr = map(_pprocess, files)
        images = [x[0] for x in arr]
        times = [x[1] for x in arr]

    images = torch.concat(images, axis=0)
    lats = np.linspace(lat_range[1], lat_range[0], images.shape[3])
    lons = np.linspace(lon_range[0], lon_range[1], images.shape[2])
    rad_image = RadarImage()
    rad_image.pytorch_image = images
    rad_image.lat_range = lat_range
    rad_image.lon_range = lon_range
    rad_image.pyart_object = files
    rad_image.grid_lat = lats
    rad_image.grid_lon = lons
    center_lat = (rad_image.lat_range[0] + rad_image.lat_range[1]) / 2.
    center_lon = (rad_image.lon_range[0] + rad_image.lon_range[1]) / 2.
    rad_image.grid_x, rad_image.grid_y = _latlon_to_xy(
        rad_image.grid_lat, rad_image.grid_lon, center_lat, center_lon)
    rad_image.times = times
    return rad_image

def _preprocess(rad_file, lat_range, lon_range):
    radar = pyart.io.read(rad_file)
    image = _render_model_input(radar, lat_range, lon_range)
    rad_time = np.datetime64(radar.time["units"].split()[2])
    del radar
    return image, rad_time

def _latlon_to_xy(lat, lon, lat0=0, lon0=0):
    R = 6371000  # Earth's radius in meters
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y
