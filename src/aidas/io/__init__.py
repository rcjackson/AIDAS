"""
============================
aidas.io (aidas.io)
============================

.. currentmodule:: aidas.io

This module handles the preprocessing of radar data.

.. autosummary::
    :toctree: generated/

    RadarImage
    preprocess_radar_image
    preprocess_radar_image_batch
    get_previous_scan
"""
from .get_radar_scan import (
    RadarImage,
    get_previous_scan,
    preprocess_radar_image,
    preprocess_radar_image_batch,
)
