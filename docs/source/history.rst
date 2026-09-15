=======
History
=======

0.6.0 (unreleased)
------------------

* Renamed the project to the Argonne AI-Driven Adaptive Sampling System (AIDAS).
  The import package is now ``aidas`` (previously ``adam``) and the distribution
  is now ``aidas-atmos`` on PyPI (previously ``adam-atmos``).
* ``adam-atmos`` is deprecated. A final transitional release forwards
  ``import adam`` to ``aidas`` with a ``DeprecationWarning``; see
  ``compat/adam-atmos/``.
* Replaced the "digital twin" framing in the project description with adaptive
  sampling / instrument tasking, which describes what the system actually does.
* Added :func:`aidas.util.aeqd_to_lonlat` and compute radar gate geolocation
  directly rather than delegating to PROJ, fixing a unit test regression across
  pyproj versions.

0.5.0 (2026-03-19)
------------------

* Added :func:`aidas.util.azimuth_from_ellipse` for deriving the instrument
  pointing direction from an ellipse fitted to the lake breeze mask.
* Added unit tests covering the new azimuth utilities.
* Expanded the notebook documentation.

0.4.0 (2026-02-09)
------------------

* Added the :mod:`aidas.triggering` module with adaptive scanning support for
  Halo Photonics lidars.
* Added the :mod:`aidas.testing` module, including a reference test dataset and
  fake SSH/SFTP clients for exercising scan triggering without hardware.
* Added docstrings and documented references for the testing and triggering
  modules.
* Allowed the user to specify their own S3 bucket when fetching radar data, and
  changed the default bucket.
* Fixed handling of ``radar_object`` when passed as either a list or an array.

0.3.0 (2025-09-15)
------------------

* Added the Sphinx-Gallery example gallery.
* Fixed the instrument pointing feature.

0.2.0 (2025-09-08)
------------------

* Added :func:`aidas.util.azimuth_point` to determine the optimal instrument
  pointing direction for adaptive scanning.
* Added unit tests for instrument pointing.
* Published the documentation to GitHub Pages.

0.1.1 (2025-07-14)
------------------

* Renamed the distribution to ``adam-atmos`` on PyPI.

0.1.0 (2025-07-14)
------------------

* First release on PyPI.
