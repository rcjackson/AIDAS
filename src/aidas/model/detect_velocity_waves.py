"""
Detect wave signatures in Doppler radial velocity by differencing consecutive scans.

The method is the one described by Miller et al. (2022), *Atmos. Meas. Tech.* **15**,
1689-1702, https://doi.org/10.5194/amt-15-1689-2022. It is not a spectral technique:
no Fourier or wavelet transform is involved. It rests on the observation that the
background wind evolves slowly compared with a propagating wave, so subtracting two
consecutive sweeps at the same elevation cancels the background and leaves the wave
perturbation behind as alternating bands of positive and negative velocity change.
"""
import logging
import os
from datetime import datetime

import numpy as np
import pyart
from scipy.ndimage import label

from ..io import RadarImage, get_previous_scan
from ..io.get_radar_scan import _nexrad_file_list
from ..util.geodesy import aeqd_to_lonlat


def _scan_time(radar):
    """
    Read the start time of a radar volume as a :func:`numpy.datetime64`.

    Parameters
    ----------
    radar: :py:meth:`pyart.core.Radar`
        The radar volume.

    Returns
    -------
    time: :func:`numpy.datetime64`
        The volume start time, with the trailing UTC designator removed because
        NumPy deprecated timezone-aware datetime64 strings.
    """
    return np.datetime64(radar.time["units"].split()[2].rstrip("Z"))


def _resolve_scan(scan, rad_time=None, bucket_name='unidata-nexrad-level2'):
    """
    Turn whatever the caller passed in into a Py-ART radar volume.

    Parameters
    ----------
    scan: str, :py:meth:`pyart.core.Radar` or :py:meth:`aidas.io.RadarImage`
        A 4-letter NEXRAD site code, a path to a radar file, a radar volume, or a
        :py:meth:`RadarImage` holding one.
    rad_time: ISO-format datestring, optional
        The date/time in YYYY-MM-DDTHH:MM:SS format wanted, used only when *scan*
        is a site code. If None, the latest scan is used.
    bucket_name: str
        The NEXRAD S3 bucket to use when *scan* is a site code.

    Returns
    -------
    radar: :py:meth:`pyart.core.Radar`
        The radar volume.
    """
    if isinstance(scan, RadarImage):
        scan = scan.pyart_object
        if isinstance(scan, (list, np.ndarray)):
            raise ValueError(
                "Wave detection differences one pair of volumes, but this RadarImage holds a "
                "batch. Pass the two scans of interest individually.")

    if isinstance(scan, pyart.core.Radar):
        return scan

    if isinstance(scan, str):
        if len(scan) == 4 and scan.isalnum() and not os.path.exists(scan):
            # A site code rather than a file: go and get the nearest volume.
            when = datetime.utcnow() if rad_time is None else datetime.strptime(rad_time, "%Y-%m-%dT%H:%M:%S")
            paths, times = _nexrad_file_list(scan, when, bucket_name=bucket_name)
            return pyart.io.read_nexrad_archive(paths[int(np.argmin(np.abs(times - when)))])
        return pyart.io.read(scan)

    raise ValueError(
        "The radar input must be a NEXRAD site code, a path to a radar file, a Py-ART radar "
        "object, or a RadarImage.")


def _select_doppler_sweep(radar, elevation, vel_field='velocity'):
    """
    Find the sweep nearest a requested elevation that actually holds velocity data.

    NEXRAD volume coverage patterns use split cuts: the lowest elevations are
    sampled twice, once in a long-PRT surveillance cut that carries reflectivity
    but no usable velocity, and again in a Doppler cut at the same fixed angle. A
    plain nearest-elevation search lands on the surveillance cut and finds an empty
    velocity field, so the sweeps are searched in order of how close they are to
    the requested elevation and the first one with unmasked velocity wins.

    Parameters
    ----------
    radar: :py:meth:`pyart.core.Radar`
        The radar volume to search.
    elevation: float
        The elevation angle wanted in degrees.
    vel_field: str
        The name of the radial velocity field.

    Returns
    -------
    sweep: int
        The index of the chosen sweep.
    """
    if vel_field not in radar.fields:
        raise ValueError(f"The radar volume has no '{vel_field}' field to detect waves in.")

    angles = np.asarray(radar.fixed_angle['data'], dtype=float)
    for sweep in np.argsort(np.abs(angles - elevation)):
        sweep = int(sweep)
        velocity = radar.get_field(sweep, vel_field)
        if np.any(~np.ma.getmaskarray(velocity)):
            logging.info(
                f"Using sweep {sweep} at {angles[sweep]:.2f} degrees elevation for wave detection.")
            return sweep

    raise ValueError(f"No sweep in this volume has any valid '{vel_field}' data.")


def _prepare_sweep(radar, elevation, reflectivity_threshold=0.0, dealias=True,
                   despeckle_size=10, vel_field='velocity', refl_field='reflectivity'):
    """
    Censor, despeckle and dealias one sweep, ready for differencing.

    This is the quality control of Miller et al. (2022): gates with reflectivity
    below 0 dBZ are discarded and speckles removed, both of which exist mainly to
    let the automated dealiasing succeed, and the velocities are then unfolded.

    The sweep is pulled out of the volume before any of that happens, so the
    dealiasing runs over one sweep rather than all twelve.

    Parameters
    ----------
    radar: :py:meth:`pyart.core.Radar`
        The radar volume.
    elevation: float
        The elevation angle wanted in degrees.
    reflectivity_threshold: float or None
        Discard velocity gates whose reflectivity is below this value in dBZ. Set
        to None to keep every gate regardless of reflectivity.
    dealias: bool
        Whether to unfold the velocities with Py-ART's region-based dealiaser.
    despeckle_size: int
        Remove contiguous velocity regions smaller than this many gates. Set to 0
        to skip despeckling.
    vel_field: str
        The name of the radial velocity field.
    refl_field: str
        The name of the reflectivity field used for censoring.

    Returns
    -------
    sweep: dict
        The prepared ``velocity`` (a masked nrays by ngates array), ``azimuth``,
        mean ``elevation``, ``range``, volume ``time``, and the index of the
        ``sweep`` it was taken from.
    """
    sweep = _select_doppler_sweep(radar, elevation, vel_field=vel_field)
    sweep_radar = radar.extract_sweeps([sweep])

    gatefilter = pyart.filters.GateFilter(sweep_radar)
    gatefilter.exclude_masked(vel_field)
    if reflectivity_threshold is not None and refl_field in sweep_radar.fields:
        reflectivity = sweep_radar.fields[refl_field]['data']
        if np.any(~np.ma.getmaskarray(reflectivity)):
            gatefilter.exclude_below(refl_field, reflectivity_threshold)
        else:
            # Some volume coverage patterns put no reflectivity on the Doppler
            # cut. Censoring on an empty field would throw the sweep away.
            logging.info(
                f"The Doppler cut carries no '{refl_field}' data, so the reflectivity censor "
                "was skipped.")
    if despeckle_size and np.any(gatefilter.gate_excluded):
        # The guard is not just an optimisation. When nothing at all has been
        # excluded, Py-ART's despeckler builds a mask that NumPy shrinks to a
        # scalar, which its own GateFilter.exclude_gates then rejects. There is
        # also nothing to do in that case: every gate holds valid data, so the
        # sweep is one contiguous object with no speckles in it.
        gatefilter = pyart.correct.despeckle_field(
            sweep_radar, vel_field, gatefilter=gatefilter, size=despeckle_size)

    if dealias:
        corrected = pyart.correct.dealias_region_based(
            sweep_radar, vel_field=vel_field, gatefilter=gatefilter)
        velocity = np.ma.masked_array(corrected['data'])
    else:
        velocity = np.ma.masked_array(sweep_radar.fields[vel_field]['data'])
    velocity = np.ma.masked_where(gatefilter.gate_excluded, velocity)

    return {'velocity': velocity,
            'azimuth': np.asarray(sweep_radar.azimuth['data'], dtype=float) % 360.0,
            'elevation': float(np.mean(sweep_radar.elevation['data'])),
            'range': np.asarray(sweep_radar.range['data'], dtype=float),
            'time': _scan_time(sweep_radar),
            'sweep': sweep}


def _nearest_azimuth_index(target_az, source_az, tolerance=1.0):
    """
    Match each target azimuth to the nearest source azimuth, wrapping through north.

    Consecutive NEXRAD volumes do not start their sweeps at the same azimuth -- the
    antenna is wherever the previous cut left it -- so ray *i* of one volume is not
    ray *i* of the next and the two sweeps cannot simply be subtracted. Matching on
    the angle itself is what makes the difference meaningful.

    Parameters
    ----------
    target_az: array-like
        The azimuths to look up, in degrees on [0, 360).
    source_az: array-like
        The azimuths to match against, in degrees on [0, 360).
    tolerance: float
        The largest acceptable angular separation in degrees. Targets whose nearest
        source is further away than this are reported invalid.

    Returns
    -------
    index: :func:`numpy.ndarray` of int
        The index into *source_az* of the nearest azimuth to each target.
    valid: :func:`numpy.ndarray` of bool
        Whether that nearest azimuth is within *tolerance*.
    """
    target_az = np.asarray(target_az, dtype=float)
    source_az = np.asarray(source_az, dtype=float)

    order = np.argsort(source_az)
    ordered = source_az[order]
    # Repeat the end points shifted by a full turn so that a target just clockwise
    # of north can match a source just anticlockwise of it.
    padded = np.concatenate(([ordered[-1] - 360.0], ordered, [ordered[0] + 360.0]))
    padded_index = np.concatenate(([order[-1]], order, [order[0]]))

    right = np.clip(np.searchsorted(padded, target_az), 1, len(padded) - 1)
    left = right - 1
    take_left = (target_az - padded[left]) <= (padded[right] - target_az)
    nearest = np.where(take_left, left, right)

    return padded_index[nearest], np.abs(target_az - padded[nearest]) <= tolerance


def _difference_sweeps(current, previous, az_tolerance=1.0):
    """
    Subtract the earlier sweep from the later one, ray by matched ray.

    Parameters
    ----------
    current: dict
        The later sweep, as returned by :func:`_prepare_sweep`.
    previous: dict
        The earlier sweep.
    az_tolerance: float
        The largest angular separation in degrees allowed when pairing rays.

    Returns
    -------
    difference: :func:`numpy.ma.MaskedArray`
        The velocity change on the current sweep's rays and gates, masked wherever
        either scan lacked data or no ray could be paired.
    """
    ngates = min(current['range'].size, previous['range'].size)
    if not np.allclose(current['range'][:ngates], previous['range'][:ngates]):
        raise ValueError(
            "The two scans have different range gate spacing, so they cannot be differenced.")

    index, valid = _nearest_azimuth_index(current['azimuth'], previous['azimuth'], az_tolerance)
    difference = current['velocity'][:, :ngates] - previous['velocity'][index][:, :ngates]
    return np.ma.masked_array(
        difference, mask=np.ma.getmaskarray(difference) | ~valid[:, np.newaxis])


def _polar_to_cartesian(field, azimuth, elevation, gate_range, grid_x, grid_y, az_tolerance=1.0):
    """
    Put a polar field onto a Cartesian grid by nearest neighbour.

    Rather than build a KD-tree over every gate, the mapping is inverted
    analytically: a grid point's ground range and azimuth give the gate whose bin
    contains it directly, which is what nearest-neighbour regridding of a polar
    field means anyway.

    Parameters
    ----------
    field: :func:`numpy.ndarray`
        The nrays by ngates field to regrid.
    azimuth: :func:`numpy.ndarray`
        The azimuth of each ray in degrees.
    elevation: float
        The mean elevation angle of the sweep in degrees.
    gate_range: :func:`numpy.ndarray`
        The range of each gate along the beam in metres.
    grid_x, grid_y: :func:`numpy.ndarray`
        The east and north coordinates of the grid in metres from the radar.
    az_tolerance: float
        The largest angular separation in degrees allowed when picking a ray.

    Returns
    -------
    gridded: :func:`numpy.ndarray`
        The field on the grid, indexed ``[y, x]``, zero wherever the grid falls
        outside the sweep.
    """
    x, y = np.meshgrid(grid_x, grid_y)

    # Gates are spaced along the beam, which is tilted, so a grid point's ground
    # range has to be lifted back onto the slant path before it indexes a gate.
    slant_range = np.hypot(x, y) / np.cos(np.radians(elevation))
    spacing = gate_range[1] - gate_range[0]
    gate = np.rint((slant_range - gate_range[0]) / spacing).astype(int)
    in_range = (gate >= 0) & (gate < field.shape[1])

    ray, ray_valid = _nearest_azimuth_index(
        (np.degrees(np.arctan2(x, y)) % 360.0).ravel(), azimuth, az_tolerance)
    ray = ray.reshape(x.shape)

    gridded = np.zeros(x.shape, dtype=field.dtype)
    inside = in_range & ray_valid.reshape(x.shape)
    gridded[inside] = field[ray[inside], gate[inside]]
    return gridded


def _remove_small_areas(mask, grid_spacing, min_area):
    """
    Drop detections that cover less area than the wave scale of interest.

    Waves exist at many scales at once, so this filter is the scale selection of
    the method as much as it is noise removal: raising *min_area* emphasises longer
    waves. Regions are eight-connected, as in the paper.

    Parameters
    ----------
    mask: :func:`numpy.ndarray`
        The binary detection mask.
    grid_spacing: float
        The grid spacing in metres.
    min_area: float
        The smallest contiguous detection to keep, in square kilometres.

    Returns
    -------
    mask: :func:`numpy.ndarray`
        The filtered mask.
    """
    min_pixels = int(np.ceil(min_area / (grid_spacing / 1000.0) ** 2))
    labels, count = label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return mask.astype(np.uint8)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # the background is not a detection
    return np.isin(labels, np.flatnonzero(sizes >= min_pixels)).astype(np.uint8)


def detect_velocity_waves(radar_scan, previous_scan=None, rad_time=None, elevation=0.5,
                          velocity_threshold=-1.0, min_area=16.0, grid_spacing=500.0,
                          reflectivity_threshold=0.0, dealias=True, despeckle_size=10,
                          az_tolerance=1.0, max_range=None, vel_field='velocity',
                          refl_field='reflectivity', bucket_name='unidata-nexrad-level2'):
    """
    Detect radial velocity wave signatures by differencing two consecutive scans.

    This implements the detection algorithm of Miller et al. (2022). Two sweeps at
    the same elevation are quality controlled, dealiased and subtracted. Because
    the background wind changes little over one volume cycle, the difference
    isolates the faster-moving wave perturbation. Only the negative half is kept,
    which emphasises the upstream convergence and the ascent implied with it, and
    the result is regridded and area filtered into a binary mask.

    The defaults are the WSR-88D values from the paper: a -1 m s-1 threshold, a
    0.5 km grid and a 16 km2 minimum area. The paper reports the result is not very
    sensitive to the exact threshold for high-amplitude waves; the minimum area
    matters more, since it selects which wave scale is emphasised.

    The mask is not self-validating. Any single convergence line -- a gust front, a
    sea breeze, a frontal boundary -- is flagged too, and those advect with the mean
    wind. A wave train shows up as several roughly parallel bands moving together,
    and confirming that they are gravity waves needs evidence beyond this mask.

    Parameters
    ----------
    radar_scan: str, :py:meth:`pyart.core.Radar` or :py:meth:`aidas.io.RadarImage`
        The later of the two scans: a 4-letter NEXRAD site code, a path to a radar
        file, a radar volume, or a :py:meth:`RadarImage`.
    previous_scan: str, :py:meth:`pyart.core.Radar` or :py:meth:`aidas.io.RadarImage`, optional
        The earlier scan, in any of the same forms. If None, the volume collected
        immediately before *radar_scan* is fetched with
        :func:`aidas.io.get_previous_scan`.
    rad_time: ISO-format datestring, optional
        The date/time in YYYY-MM-DDTHH:MM:SS format wanted, used only when
        *radar_scan* is a site code. If None, the latest scan is used.
    elevation: float
        The elevation angle to detect waves at in degrees. The nearest sweep
        carrying velocity data is used.
    velocity_threshold: float
        Flag gates where the velocity change is below this value in m s-1. The
        paper's WSR-88D value is -1.
    min_area: float
        The smallest contiguous detection to keep, in square kilometres.
    grid_spacing: float
        The spacing of the output Cartesian grid in metres.
    reflectivity_threshold: float or None
        Discard velocity gates whose reflectivity is below this value in dBZ. Set
        to None to keep every gate regardless of reflectivity.
    dealias: bool
        Whether to unfold the velocities before differencing.
    despeckle_size: int
        Remove contiguous velocity regions smaller than this many gates. Set to 0
        to skip despeckling.
    az_tolerance: float
        The largest angular separation in degrees allowed when pairing rays between
        the two scans.
    max_range: float, optional
        The half-width of the output grid in metres. If None, it reaches the
        furthest gate with valid velocity, which keeps the grid off the empty
        ground beyond the Doppler cut's unambiguous range.
    vel_field: str
        The name of the radial velocity field.
    refl_field: str
        The name of the reflectivity field used for censoring.
    bucket_name: str
        The NEXRAD S3 bucket to use when a scan has to be fetched.

    Returns
    -------
    radar_scan: :py:meth:`aidas.io.RadarImage`
        The :py:meth:`RadarImage` with ``velocity_wave_mask`` and the grid it lives
        on set. A new :py:meth:`RadarImage` is made if one was not passed in.

    References
    ----------
    Miller, M. A., Oue, M., Kumjian, M. R., and Kollias, P., 2022: A method for
    detecting atmospheric gravity waves using Doppler radial velocity.
    *Atmos. Meas. Tech.*, **15**, 1689-1702, https://doi.org/10.5194/amt-15-1689-2022.
    """
    rad_image = radar_scan if isinstance(radar_scan, RadarImage) else RadarImage()
    current_radar = _resolve_scan(radar_scan, rad_time=rad_time, bucket_name=bucket_name)
    if previous_scan is None:
        previous_radar = get_previous_scan(current_radar, bucket_name=bucket_name)
    else:
        previous_radar = _resolve_scan(previous_scan, bucket_name=bucket_name)

    prepare = dict(reflectivity_threshold=reflectivity_threshold, dealias=dealias,
                   despeckle_size=despeckle_size, vel_field=vel_field, refl_field=refl_field)
    current = _prepare_sweep(current_radar, elevation, **prepare)
    previous = _prepare_sweep(previous_radar, elevation, **prepare)

    if previous['time'] > current['time']:
        logging.warning(
            "The scan given as previous was collected after the current one; swapping them so "
            "the difference still runs forwards in time.")
        current, previous = previous, current
        current_radar = previous_radar

    difference = _difference_sweeps(current, previous, az_tolerance=az_tolerance)
    # Keep the negative half only: the paper's convergence and implied ascent.
    detections = np.ma.filled(difference < velocity_threshold, False)

    if max_range is None:
        valid_gates = np.flatnonzero(np.any(~np.ma.getmaskarray(current['velocity']), axis=0))
        if valid_gates.size == 0:
            raise ValueError("The chosen sweep has no valid velocity data left after censoring.")
        max_range = current['range'][valid_gates[-1]] * np.cos(np.radians(current['elevation']))

    edge = int(np.floor(max_range / grid_spacing)) * grid_spacing
    grid_x = np.arange(-edge, edge + grid_spacing, grid_spacing)
    grid_y = grid_x.copy()

    mask = _polar_to_cartesian(detections, current['azimuth'], current['elevation'],
                               current['range'][:detections.shape[1]], grid_x, grid_y,
                               az_tolerance=az_tolerance)
    mask = _remove_small_areas(mask, grid_spacing, min_area)

    x, y = np.meshgrid(grid_x, grid_y)
    lon, lat = aeqd_to_lonlat(x, y, float(current_radar.longitude['data'][0]),
                              float(current_radar.latitude['data'][0]))

    rad_image.velocity_wave_mask = mask
    rad_image.wave_grid_x = grid_x
    rad_image.wave_grid_y = grid_y
    rad_image.wave_grid_lat = lat
    rad_image.wave_grid_lon = lon
    rad_image.wave_scan_times = (previous['time'], current['time'])
    rad_image.wave_sweep = current['sweep']
    if rad_image.pyart_object is None:
        rad_image.pyart_object = current_radar
    if rad_image.times is None:
        rad_image.times = [current['time']]
    return rad_image
