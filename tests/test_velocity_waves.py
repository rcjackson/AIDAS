#!/usr/bin/env python
"""Tests for the radial velocity wave detection of `aidas`."""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyart
import pytest
from scipy.ndimage import label

import aidas
from aidas.io import RadarImage
from aidas.io.get_radar_scan import _nexrad_file_list
from aidas.model.detect_velocity_waves import (
    _difference_sweeps,
    _nearest_azimuth_index,
    _prepare_sweep,
    _remove_small_areas,
    _select_doppler_sweep,
)

matplotlib.use('Agg')

WAVELENGTH_KM = 20.0
GATE_SPACING = 500.0
NGATES = 200


def _synthetic_volume(shift_km=0.0, start_az=0.0, amplitude=3.0, background=10.0,
                      wavelength_km=WAVELENGTH_KM, nrays=360,
                      scan_time='2025-07-15T18:00:00', split_cut=True, nyquist=40.0,
                      fold=False):
    """
    Build a PPI volume carrying a plane wave in radial velocity.

    The wave crests run north-south and the wave travels east, so shifting it by
    *shift_km* between two volumes is what the detector is meant to pick up. A
    constant *background* is added to both volumes to check that it cancels in the
    difference, which is the assumption the whole method rests on.

    When *split_cut* is set the volume mimics a NEXRAD split cut: sweep 0 is a
    surveillance cut with reflectivity but no velocity, and sweep 1 is the Doppler
    cut at the same elevation.
    """
    radar = pyart.testing.make_empty_ppi_radar(NGATES, nrays, 2)
    radar.range['data'] = np.arange(NGATES, dtype=float) * GATE_SPACING + GATE_SPACING / 2
    azimuth = (np.arange(nrays) * (360.0 / nrays) + start_az) % 360
    radar.azimuth['data'] = np.concatenate([azimuth, azimuth])
    radar.elevation['data'] = np.full(2 * nrays, 0.5)
    radar.fixed_angle['data'] = np.array([0.5, 0.5])
    # Ray times of a split cut: the surveillance sweep first, the Doppler sweep
    # after it, both offset from the start of the volume.
    radar.time['data'] = np.concatenate([np.linspace(0.0, 30.0, nrays),
                                         np.linspace(32.0, 57.0, nrays)])
    radar.latitude['data'] = np.array([41.6])
    radar.longitude['data'] = np.array([-88.0])
    radar.altitude['data'] = np.array([200.0])
    radar.time['units'] = f'seconds since {scan_time}Z'
    radar.instrument_parameters = {'nyquist_velocity': {'data': np.full(2 * nrays, nyquist)}}

    east = radar.range['data'][np.newaxis, :] * np.sin(np.radians(azimuth))[:, np.newaxis]
    wave = background + amplitude * np.sin(
        2 * np.pi * (east / 1000.0 - shift_km) / wavelength_km)
    if fold:
        # Wrap the velocities into the Nyquist interval, as the radar itself would.
        wave = ((wave + nyquist) % (2 * nyquist)) - nyquist

    velocity = np.ma.masked_array(np.concatenate([wave, wave]),
                                  mask=np.zeros((2 * nrays, NGATES), dtype=bool))
    if split_cut:
        velocity.mask[:nrays] = True
    radar.add_field('velocity', {'data': velocity, 'units': 'm/s', '_FillValue': -9999.0})
    radar.add_field('reflectivity',
                    {'data': np.ma.masked_array(np.full((2 * nrays, NGATES), 20.0)),
                     'units': 'dBZ', '_FillValue': -9999.0})
    return radar


def _wave_pair(shift_km=5.0, start_az=0.0, **kwargs):
    """Two synthetic volumes five minutes apart, the wave shifted between them."""
    previous = _synthetic_volume(shift_km=0.0, scan_time='2025-07-15T18:00:00', **kwargs)
    current = _synthetic_volume(shift_km=shift_km, start_az=start_az,
                                scan_time='2025-07-15T18:05:00', **kwargs)
    return current, previous


def test_split_cut_sweep_selection():
    """The Doppler cut is chosen even though the surveillance cut is listed first."""
    radar = _synthetic_volume()
    assert _select_doppler_sweep(radar, 0.5) == 1

    no_velocity = _synthetic_volume()
    no_velocity.fields['velocity']['data'].mask[:] = True
    with pytest.raises(ValueError):
        _select_doppler_sweep(no_velocity, 0.5)

    with pytest.raises(ValueError):
        _select_doppler_sweep(_synthetic_volume(), 0.5, vel_field='not_a_field')


def test_detects_plane_wave():
    """A travelling plane wave is detected as bands one wavelength apart."""
    current, previous = _wave_pair()
    scan = aidas.model.detect_velocity_waves(current, previous, dealias=False)

    mask = scan.velocity_wave_mask
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1}
    assert mask.sum() > 0

    # The grid is square, centred on the radar, and reaches the last valid gate.
    assert mask.shape == (scan.wave_grid_y.size, scan.wave_grid_x.size)
    assert scan.wave_grid_x[1] - scan.wave_grid_x[0] == 500.0
    np.testing.assert_allclose(scan.wave_grid_x[-1], NGATES * GATE_SPACING - GATE_SPACING / 2,
                               atol=GATE_SPACING)
    assert scan.wave_grid_lat.shape == mask.shape
    # The times are those of the Doppler sweep, 32 s into each volume, not of the
    # volumes themselves.
    assert scan.wave_scan_times == (np.datetime64('2025-07-15T18:00:32'),
                                    np.datetime64('2025-07-15T18:05:32'))

    # The crests run north-south, so a line of longitude crosses one band per
    # wavelength. Measure the spacing of the band edges along the centre row.
    centre_row = mask[mask.shape[0] // 2]
    starts = scan.wave_grid_x[np.flatnonzero(np.diff(centre_row.astype(int)) == 1)]
    assert starts.size >= 5
    np.testing.assert_allclose(np.diff(starts) / 1000.0, WAVELENGTH_KM, atol=1.0)


def test_stationary_field_gives_no_detections():
    """With nothing moving the difference is zero, so there is nothing to detect."""
    current = _synthetic_volume(scan_time='2025-07-15T18:05:00')
    previous = _synthetic_volume(scan_time='2025-07-15T18:00:00')
    scan = aidas.model.detect_velocity_waves(current, previous, dealias=False)
    assert scan.velocity_wave_mask.sum() == 0


def test_background_wind_cancels():
    """A strong uniform background does not change the answer: it subtracts out."""
    calm = aidas.model.detect_velocity_waves(*_wave_pair(background=0.0), dealias=False)
    windy = aidas.model.detect_velocity_waves(*_wave_pair(background=40.0), dealias=False)
    np.testing.assert_array_equal(calm.velocity_wave_mask, windy.velocity_wave_mask)


def test_rays_are_matched_by_azimuth():
    """
    The answer does not depend on where the sweep happened to start.

    Consecutive NEXRAD volumes begin their sweeps at different azimuths, so a
    detector that subtracted ray by ray would give a different mask here.
    """
    aligned = aidas.model.detect_velocity_waves(*_wave_pair(start_az=0.0), dealias=False)
    rotated = aidas.model.detect_velocity_waves(*_wave_pair(start_az=17.3), dealias=False)

    def band_starts(scan):
        row = scan.velocity_wave_mask[scan.velocity_wave_mask.shape[0] // 2]
        return scan.wave_grid_x[np.flatnonzero(np.diff(row.astype(int)) == 1)]

    # The bands land in the same places. Sampling the wave at azimuths offset by up
    # to half a ray spacing moves a band edge by a grid cell here and there, so the
    # masks are not identical pixel for pixel, but the wave they describe is.
    np.testing.assert_allclose(band_starts(aligned), band_starts(rotated), atol=1000.0)
    agreement = (aligned.velocity_wave_mask == rotated.velocity_wave_mask).mean()
    assert agreement > 0.97, f"masks agree on only {agreement:.3f} of the grid"


def test_nearest_azimuth_index_wraps_through_north():
    """Matching is circular, and targets too far from any ray are flagged."""
    source = np.array([0.5, 90.0, 180.0, 359.5])
    index, valid = _nearest_azimuth_index(np.array([0.2, 359.9, 90.4, 270.0]), source,
                                          tolerance=1.0)
    # 359.5 is nearer to 270 than 180 is, but not within the tolerance.
    np.testing.assert_array_equal(index, [0, 3, 1, 3])
    np.testing.assert_array_equal(valid, [True, True, True, False])

    # Without wrapping, a target just short of north would match the ray at 270
    # rather than the one just past north.
    index, valid = _nearest_azimuth_index(np.array([359.9]),
                                          np.array([1.0, 90.0, 180.0, 270.0]), tolerance=2.0)
    np.testing.assert_array_equal(index, [0])
    np.testing.assert_array_equal(valid, [True])


def test_threshold_controls_sensitivity():
    """A stricter velocity threshold detects less of the wave."""
    current, previous = _wave_pair()
    loose = aidas.model.detect_velocity_waves(
        current, previous, velocity_threshold=-0.5, dealias=False)
    strict = aidas.model.detect_velocity_waves(
        current, previous, velocity_threshold=-3.0, dealias=False)
    assert strict.velocity_wave_mask.sum() < loose.velocity_wave_mask.sum()


def test_minimum_area_filter():
    """Eight-connected regions below the minimum area are dropped, larger ones kept."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1          # 20 x 20 cells of 0.25 km2 = 100 km2
    mask[60:62, 60:62] = 1          # 2 x 2 cells = 1 km2
    mask[80, 80] = 1                # a single cell speckle
    mask[90, 90] = 1
    mask[91, 91] = 1                # touches the one above only at a corner

    kept = _remove_small_areas(mask, 500.0, 16.0)
    assert kept[10:30, 10:30].all()
    assert kept.sum() == 400

    # Eight-connectivity: the two corner-touching cells are one region of 0.5 km2.
    kept = _remove_small_areas(mask, 500.0, 0.4)
    assert kept[90, 90] == 1 and kept[91, 91] == 1
    assert kept[80, 80] == 0


def test_grid_spacing_and_max_range_are_honoured():
    """The output grid follows the spacing and range asked for."""
    current, previous = _wave_pair()
    scan = aidas.model.detect_velocity_waves(
        current, previous, grid_spacing=1000.0, max_range=50000.0,
        min_area=16.0, dealias=False)
    assert scan.wave_grid_x[1] - scan.wave_grid_x[0] == 1000.0
    assert scan.wave_grid_x[-1] == 50000.0
    assert scan.velocity_wave_mask.shape == (101, 101)


def test_dealiasing_runs():
    """The dealiasing path works on data that needs no unfolding."""
    current, previous = _wave_pair()
    scan = aidas.model.detect_velocity_waves(current, previous, dealias=True)
    assert scan.velocity_wave_mask.sum() > 0


def test_reflectivity_censoring():
    """Gates below the reflectivity threshold are thrown away before differencing."""
    current, previous = _wave_pair()
    for radar in (current, previous):
        radar.fields['reflectivity']['data'][:] = -10.0

    censored = _prepare_sweep(current, 0.5, reflectivity_threshold=0.0, dealias=False)
    assert np.ma.getmaskarray(censored['velocity']).all()

    kept = _prepare_sweep(current, 0.5, reflectivity_threshold=None, dealias=False)
    assert not np.ma.getmaskarray(kept['velocity']).all()


def test_mismatched_gate_spacing_is_rejected():
    """Two scans on different range grids cannot be differenced."""
    current, previous = _wave_pair()
    previous.range['data'] = previous.range['data'] * 2
    with pytest.raises(ValueError, match="range gate spacing"):
        _difference_sweeps(_prepare_sweep(current, 0.5, dealias=False),
                           _prepare_sweep(previous, 0.5, dealias=False))


def test_scans_out_of_order_are_swapped():
    """Passing the scans the wrong way round still differences forwards in time."""
    current, previous = _wave_pair()
    forwards = aidas.model.detect_velocity_waves(current, previous, dealias=False)
    backwards = aidas.model.detect_velocity_waves(previous, current, dealias=False)
    np.testing.assert_array_equal(forwards.velocity_wave_mask, backwards.velocity_wave_mask)
    assert backwards.wave_scan_times == forwards.wave_scan_times


def test_batch_radar_image_is_rejected():
    """Wave detection needs one pair of volumes, not a batch."""
    batch = RadarImage()
    batch.pyart_object = ['one.nc', 'two.nc']
    with pytest.raises(ValueError, match="batch"):
        aidas.model.detect_velocity_waves(batch, _synthetic_volume())


def test_pointing_from_the_wave_mask():
    """The lidar can be steered from the wave mask, not just the lake breeze."""
    current, previous = _wave_pair()
    scan = aidas.model.detect_velocity_waves(current, previous, dealias=False)

    radar_lat = float(current.latitude['data'][0])
    radar_lon = float(current.longitude['data'][0])
    azimuth, lat, lon, dist = aidas.util.azimuth_point(
        radar_lon, radar_lat, scan, mask='velocity_wave')

    assert 0 <= azimuth < 360
    assert scan.wave_grid_lat.min() <= lat <= scan.wave_grid_lat.max()
    assert scan.wave_grid_lon.min() <= lon <= scan.wave_grid_lon.max()
    # The instrument sits at the radar, inside the wave field, so the nearest
    # detection is closer than the grid is wide.
    assert 0 <= dist < scan.wave_grid_x[-1]

    # The bands run north-south, so the ellipse fitted to them has a north-south
    # major axis and the pointing direction is east or west.
    ellipse_azimuth = aidas.util.azimuth_from_ellipse(
        radar_lon, radar_lat, scan, mask='velocity_wave')[0]
    assert abs(np.cos(np.radians(ellipse_azimuth))) < 0.2, (
        f"expected an east-west azimuth across the bands, got {ellipse_azimuth:.1f}")


def test_unknown_mask_name_is_rejected():
    current, previous = _wave_pair()
    scan = aidas.model.detect_velocity_waves(current, previous, dealias=False)
    with pytest.raises(ValueError, match="not a mask"):
        aidas.util.azimuth_point(-88.0, 41.6, scan, mask='nonsense')


def test_pointing_without_a_wave_mask_is_rejected():
    with pytest.raises(ValueError, match="detect_velocity_waves"):
        aidas.util.azimuth_point(-88.0, 41.6, RadarImage(), mask='velocity_wave')


def test_visualize_without_a_mask_is_rejected():
    with pytest.raises(ValueError, match="detect_velocity_waves"):
        aidas.vis.visualize_velocity_waves(RadarImage())


def test_detect_velocity_waves_on_nexrad():
    """
    End to end on real KLOT data, fetching the previous volume itself.

    15 July 2025 is a fair weather lake breeze day rather than a wave event, so the
    point of this test is that the plumbing works on real split-cut data: the
    Doppler cut is found, the volumes are paired, and the area filter clears the
    clear-air speckle out of the mask.
    """
    scan = aidas.model.detect_velocity_waves('KLOT', rad_time='2025-07-15T18:13:45')
    assert scan.wave_scan_times == (np.datetime64('2025-07-15T18:08:00'),
                                    np.datetime64('2025-07-15T18:14:59'))
    assert scan.velocity_wave_mask.shape == (1181, 1181)
    np.testing.assert_almost_equal(scan.velocity_wave_mask.sum(), 1569, decimal=-2)

    # Every surviving region is at least the 16 km2 the filter was asked for.
    sizes = np.bincount(label(scan.velocity_wave_mask, structure=np.ones((3, 3)))[0].ravel())[1:]
    assert (sizes * 0.25 >= 16.0).all()


def test_kokx_winter_storm_case():
    """
    The 1 February 2021 KOKX case of the paper, its Fig. 5.

    Asking for the 09:03:09 volume must difference it against the 08:56 one, as the
    paper does. This volume runs SAILS, so it holds four sweeps at 0.48 degrees --
    the base split cut and a supplemental one taken four minutes later. Picking the
    supplemental cut of one volume against the base cut of the other would compare
    sweeps minutes out of step, so the base cut has to win both times.
    """
    scan = aidas.model.detect_velocity_waves('KOKX', rad_time='2021-02-01T09:03:09')

    assert scan.wave_sweep == 1, "the base Doppler cut, not the SAILS supplemental cut"
    earlier, later = scan.wave_scan_times
    assert earlier == np.datetime64('2021-02-01T08:56:35')
    assert later == np.datetime64('2021-02-01T09:03:41')
    # One volume cycle apart, not one cycle plus the SAILS offset.
    assert (later - earlier) / np.timedelta64(1, 's') == 426

    # The paper reads this case as banded convergence in the rain and snow rather
    # than as waves, but either way it is a widespread, strongly banded signal.
    mask = scan.velocity_wave_mask
    np.testing.assert_almost_equal(mask.sum(), 81973, decimal=-3)
    assert 0.02 < mask.mean() < 0.12


def test_kokx_wave_case_matches_the_paper():
    """
    The 26 December 2010 KOKX case the paper demonstrates the method with, its Fig. 2.

    This is the one real case with published wave properties to check against: long
    axes running SSW to NNE, a horizontal wavelength on the order of 12 to 18 km,
    and a train propagating northwest. The volumes are the older V03 files, whose
    names the S3 listing has to recognise for the pairing to work at all.
    """
    scan = aidas.model.detect_velocity_waves('KOKX', rad_time='2010-12-26T23:45:15',
                                             max_range=137000.)
    # The paper quotes these two sweeps as 23:40:00 and 23:45:47.
    assert scan.wave_scan_times == (np.datetime64('2010-12-26T23:40:00'),
                                    np.datetime64('2010-12-26T23:45:48'))
    assert scan.wave_grid_x[-1] == 137000.

    # Recover the dominant wave from the mask itself and compare with the paper.
    mask = scan.velocity_wave_mask.astype(float)
    mask = mask - mask.mean()
    window = np.outer(np.hanning(mask.shape[0]), np.hanning(mask.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(mask * window))) ** 2

    frequency = np.fft.fftshift(np.fft.fftfreq(mask.shape[0], d=0.5))  # cycles per km
    fx, fy = np.meshgrid(frequency, frequency)
    scale = np.hypot(fx, fy)
    # Ignore the shape of the echo area itself and the smallest resolvable scales.
    power[(scale < 1 / 40.) | (scale > 1 / 4.)] = 0
    peak = np.unravel_index(np.argmax(power), power.shape)

    wavelength = 1 / np.hypot(fx[peak], fy[peak])
    assert 10.0 < wavelength < 22.0, f"wavelength {wavelength:.1f} km is not the paper's 12-18 km"

    # The wavevector points across the bands, so the long axis is perpendicular to
    # it. SSW to NNE is around 20 to 30 degrees.
    long_axis = (np.degrees(np.arctan2(fx[peak], fy[peak])) + 90) % 180
    assert 15.0 < long_axis < 60.0, f"band axis {long_axis:.0f} deg is not SSW to NNE"


def test_get_previous_scan():
    """The previous volume is the one before, not the nearest one before the clock."""
    previous = aidas.io.get_previous_scan('KLOT', '2025-07-15T18:13:45')
    assert previous.time['units'].split()[2] == '2025-07-15T18:06:46Z'

    # Asking from part way through a volume resolves that volume first, then steps
    # back, so the answer is the same.
    previous = aidas.io.get_previous_scan('KLOT', '2025-07-15T18:13:00')
    assert previous.time['units'].split()[2] == '2025-07-15T18:06:46Z'


def test_volume_listing_skips_metadata_files():
    """
    Only real volumes are listed, in ascending time order.

    Some volumes have an _MDM metadata file beside them carrying the same
    timestamp. Py-ART cannot read one, and because the timestamps match it would be
    picked whenever that time was the closest, so it has to be left out. Older
    volumes are named _V03.gz rather than _V06 and do have to be kept.
    """
    paths, times = _nexrad_file_list('KOKX', datetime(2021, 2, 1, 9, 3, 9))
    assert not any(path.endswith('_MDM') for path in paths)
    assert all(later >= earlier for earlier, later in zip(times[:-1], times[1:]))
    assert 's3://unidata-nexrad-level2/2021/02/01/KOKX/KOKX20210201_090309_V06' in paths

    paths, _ = _nexrad_file_list('KOKX', datetime(2010, 12, 26, 23, 45, 15))
    assert any(path.endswith('KOKX20101226_234515_V03.gz') for path in paths)


def test_plot_velocity_wave_mask_alone():
    """The mask can be drawn on its own, into an axes the caller supplies."""
    import cartopy.crs as ccrs

    current, previous = _wave_pair()
    scan = aidas.model.detect_velocity_waves(current, previous, dealias=False)

    fig, ax = aidas.vis.plot_velocity_wave_mask(scan)
    assert 'Wave mask' in ax.get_title()

    # Into a caller's axes, which is how several settings get compared side by side.
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection=ccrs.PlateCarree()))
    for axis, min_area in zip(axes, (1.0, 400.0)):
        tuned = aidas.model.detect_velocity_waves(
            current, previous, dealias=False, min_area=min_area)
        returned_fig, returned_ax = aidas.vis.plot_velocity_wave_mask(
            tuned, ax=axis, title=f"{min_area:g}")
        assert returned_ax is axis
        assert returned_fig is fig
        assert returned_ax.get_title() == f"{min_area:g}"

    with pytest.raises(ValueError, match="detect_velocity_waves"):
        aidas.vis.plot_velocity_wave_mask(RadarImage())


@pytest.mark.mpl_image_compare(tolerance=50)
def test_visualize_velocity_waves():
    """The mask gets a panel of its own, beside the velocity it came from."""
    current, previous = _wave_pair()
    scan = aidas.model.detect_velocity_waves(current, previous, dealias=False)
    fig, axes = aidas.vis.visualize_velocity_waves(scan)
    assert fig is not None
    assert len(axes) == 2
    assert 'Wave mask' in axes[1].get_title()
    return fig


def test_the_plotted_velocity_is_the_one_detected_from():
    """
    The sweep kept for plotting holds the field the mask was computed from.

    It has to be the dealiased, censored velocity rather than the raw folded field
    in the original volume, or the picture and the mask would disagree.
    """
    current, previous = _wave_pair()
    scan = aidas.model.detect_velocity_waves(current, previous, dealias=False,
                                             reflectivity_threshold=None)

    assert scan.wave_sweep_radar is not None
    assert scan.wave_sweep_radar.nsweeps == 1
    kept = scan.wave_sweep_radar.fields['velocity']['data']
    # The Doppler cut of the original volume, which is sweep 1 of a split cut.
    original = current.extract_sweeps([scan.wave_sweep]).fields['velocity']['data']
    np.testing.assert_allclose(np.ma.compressed(kept), np.ma.compressed(original))

    # The caller's volume still holds every sweep, untouched.
    assert current.nsweeps == 2


def test_velocity_is_dealiased_before_the_mask_is_made():
    """
    Folded velocities are unfolded first, as the paper specifies.

    The wave here swings further than the Nyquist velocity, so the radar folds it
    and the fold edges become huge apparent velocity changes. Detecting on the
    folded field would mark those edges instead of the wave.
    """
    kwargs = dict(amplitude=12.0, background=0.0, nyquist=5.0)
    folded_current, folded_previous = _wave_pair(fold=True, **kwargs)
    true_current, true_previous = _wave_pair(fold=False, **kwargs)

    folded = folded_current.extract_sweeps([1]).fields['velocity']['data']
    assert np.abs(folded).max() <= 5.0, "the test data should be folded to start with"

    dealiased = aidas.model.detect_velocity_waves(folded_current, folded_previous,
                                                  dealias=True)
    kept = dealiased.wave_sweep_radar.fields['velocity']['data']
    assert np.abs(kept).max() > 5.0, "the velocities were never unfolded"

    # Detecting on the unfolded field recovers the same bands as detecting on the
    # field that was never folded; leaving it folded does not.
    truth = aidas.model.detect_velocity_waves(true_current, true_previous, dealias=False)
    still_folded = aidas.model.detect_velocity_waves(folded_current, folded_previous,
                                                     dealias=False)
    agreement = (dealiased.velocity_wave_mask == truth.velocity_wave_mask).mean()
    folded_agreement = (still_folded.velocity_wave_mask == truth.velocity_wave_mask).mean()
    assert agreement > folded_agreement, (
        f"dealiasing did not help: {agreement:.3f} against {folded_agreement:.3f}")
