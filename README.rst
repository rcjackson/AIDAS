==========================================
Argonne AI-Driven Adaptive Sampling System
==========================================


.. image:: https://img.shields.io/pypi/v/aidas-atmos.svg
        :target: https://pypi.org/project/aidas-atmos/
        :alt: PyPI version

.. image:: https://github.com/rcjackson/AIDAS/actions/workflows/unit-tests.yml/badge.svg
        :target: https://github.com/rcjackson/AIDAS/actions/workflows/unit-tests.yml
        :alt: Unit tests

.. image:: https://img.shields.io/badge/docs-latest-blue.svg
        :target: https://rcjackson.github.io/AIDAS/
        :alt: Documentation


AIDAS is the Argonne AI-Driven Adaptive Sampling System, a Python package for
AI-guided instrument tasking over the Argonne Testbed for Multiscale Observational
Studies (ATMOS). It detects mesoscale boundaries in NEXRAD radar data with deep
learning models and uses those detections to cue ground-based instruments --
closing the loop from radar observation to lidar scan strategy.

The current release contains a model that determines the lake breeze front location
from the 0.5 degree scan of the NEXRAD radar, utilities for deriving an optimal
instrument pointing direction from the resulting mask, and scan triggering for
Halo Photonics Doppler lidars.

Installation
------------

The recommended way to install AIDAS is via pip. This will ensure you get the latest
stable release and all required dependencies:

.. code-block:: console

        pip install aidas-atmos

Getting Started
---------------

After installation, you can import AIDAS in your Python scripts or notebooks:

.. code-block:: python

        import aidas

See the documentation and example notebooks for usage details.

Features
--------

* Lake breeze detection from NEXRAD radar data using deep learning models.
* Easy-to-use API for loading and processing radar data.
* Instrument pointing utilities for adaptive sampling.
* Scan triggering for Halo Photonics Doppler lidars.
* Example notebooks demonstrating functionality.

Links
-----

* Documentation: https://rcjackson.github.io/AIDAS/
* Source code: https://github.com/rcjackson/AIDAS
* Free software: BSD license

Project history
---------------

AIDAS was previously released on PyPI as ``adam-atmos`` (the ATMOS Analogue Digital
Twin, or ADAM) through version 0.5.0. The ``adam-atmos`` distribution is deprecated;
see `Installation <https://rcjackson.github.io/AIDAS/installation.html>`_ for
migration notes.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
