Detecting waves in radial velocity
==================================

Alongside the lake breeze model, AIDAS can detect wave signatures in Doppler radial
velocity. This is not a machine learning product: it is the algorithm of
`Miller et al. (2022) <https://doi.org/10.5194/amt-15-1689-2022>`_, and it works by
subtracting two consecutive radar volumes from one another.

The idea is that the background wind changes very little over one volume cycle,
while a propagating wave moves a long way in the same time. Subtracting the two
sweeps therefore cancels the background and leaves the wave behind, as bands of
alternating velocity change. AIDAS keeps the negative half of that difference, which
marks the upstream convergence and the ascent implied with it, and turns it into a
binary mask.

Running the detector
--------------------

The detector needs two scans. If you only give it one, it fetches the volume before
it for you:

.. code-block:: python

    import aidas

    rad_scan = aidas.model.detect_velocity_waves('KLOT', rad_time='2025-07-15T18:13:45')

You can also hand it both scans yourself, as Py-ART radar objects, file paths, or a
:py:meth:`RadarImage` you have already preprocessed:

.. code-block:: python

    rad_scan = aidas.model.detect_velocity_waves(later_scan, earlier_scan)

The result carries ``velocity_wave_mask``, where 1 marks a detection, along with the
grid it sits on in ``wave_grid_lat`` and ``wave_grid_lon``, and the two scan times
that were differenced in ``wave_scan_times``. Note that the wave mask lives on its
own Cartesian grid centred on the radar, at 0.5 km spacing by default, rather than on
the 256 by 256 domain grid the lake breeze mask uses.

To see it:

.. code-block:: python

    aidas.vis.visualize_velocity_waves(rad_scan)

That particular example is a fair weather afternoon over Chicago rather than a wave
event, and the mask comes back nearly empty, which is the right answer. The clear-air
velocity field near the radar is noisy and the area filter throws nearly all of that
noise away; what survives sits in the precipitation to the south, as a handful of
convergence lines.

The case from the paper
-----------------------

For a case with waves in it, here is the one the paper demonstrates the method with:
KOKX at Upton, New York, late on 26 December 2010, with a low centre a couple of
hundred kilometres to the southeast.

.. plot::

    import aidas
    import matplotlib.pyplot as plt

    rad_scan = aidas.model.detect_velocity_waves(
        'KOKX', rad_time='2010-12-26T23:45:15', max_range=137000.)
    fig, ax = aidas.vis.visualize_velocity_waves(rad_scan, vmin=-25, vmax=25)
    plt.show()

Taking a two dimensional Fourier transform of the mask puts the dominant sets at
wavelengths of 14 to 19 km with their long axes running 28 to 39 degrees, SSW to NNE,
and the wave train moving off to the northwest. The paper reports wavelengths on the
order of 12 to 18 km with SSW to NNE axes for the same case. Several roughly parallel
bands moving together like this are what a wave train looks like, and what
distinguishes it from a single boundary.

Tuning the detection
--------------------

The defaults are the WSR-88D values from the paper: flag velocity changes below
-1 m s\ :sup:`-1`, grid at 0.5 km, and throw away detections covering less than
16 km\ :sup:`2`.

.. code-block:: python

    rad_scan = aidas.model.detect_velocity_waves(
        'KLOT', rad_time='2025-07-15T18:13:45',
        velocity_threshold=-1.0, grid_spacing=500., min_area=16.)

Of the three, ``min_area`` is the one worth thinking about. Waves exist at many
scales at once, so the area filter is not only removing noise -- it is choosing which
wave scale you see. Raising it emphasises longer waves. The paper reports that the
result is not very sensitive to the exact velocity threshold for high-amplitude
waves, and thresholds for other instruments should scale with their noise floor.

What the mask can and cannot tell you
-------------------------------------

Two limits are worth keeping in mind.

The first is sampling. A wave has to move at least one resolution volume between
scans to show up at all, and it has to move less than half a wavelength for its speed
and direction to come out right:

.. math::

    t < \frac{\lambda}{2V}

With a roughly four minute volume cycle, a 10 km wave has to be travelling slower
than about 21 m s\ :sup:`-1`. Faster than that and the wave train appears to
stand still, flash in place, or move backwards. Clear-air volume coverage patterns
can take ten minutes to come round again, which makes the limit tighter still.
``wave_scan_times`` holds the times of the two sweeps that were actually
differenced, so the interval can be checked rather than assumed. A pattern running
SAILS revisits the lowest elevation part way through each volume, but AIDAS takes the
base cut from both volumes, so the interval is one full volume cycle.

The second is interpretation. Any linear convergence feature is flagged, not just
waves: gust fronts, sea and lake breezes, fronts, and terrain effects all produce a
band. Those advect with the mean wind. A wave train instead looks like several
roughly parallel bands moving together, and may move independently of the prevailing
flow. Confirming that such a train really is a gravity wave takes evidence this mask
does not carry.

The paper makes that point with a second KOKX case, the winter storm of 1 February
2021, which AIDAS will happily detect:

.. code-block:: python

    rad_scan = aidas.model.detect_velocity_waves('KOKX', rad_time='2021-02-01T09:03:09')

It produces strong banding in both the rain and the snow, but the bands stay locked
to the precipitation structures and move with them, so the paper attributes them to
mesoscale convergence rather than to waves. The mask cannot tell the two apart; that
is the reader's job.

Pointing an instrument at a wave train
--------------------------------------

The pointing utilities take the wave mask as readily as the lake breeze mask:

.. code-block:: python

    angle, lat, lon, dist = aidas.util.azimuth_point(
        atmos_lon, atmos_lat, rad_scan, mask='velocity_wave')

:func:`aidas.util.azimuth_from_ellipse` is often the better choice here, since a wave
train of parallel bands fits an ellipse far more naturally than a single front does,
and it returns an azimuth across the bands rather than along them.
