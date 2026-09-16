"""
============================
aidas.model (aidas.model)
============================

.. currentmodule:: aidas.model

This module handles the detection of mesoscale features in radar data: the lake
breeze, inferred with a fine-tuned ResNet50, and radial velocity wave signatures,
detected by differencing consecutive scans.

.. autosummary::
    :toctree: generated/

    infer_lake_breeze
    infer_lake_breeze_batch
    detect_velocity_waves
"""
from .detect_velocity_waves import detect_velocity_waves
from .predict_lake_breeze import infer_lake_breeze, infer_lake_breeze_batch
