""" 
==============================
AIDAS Triggering Module
==============================
.. currentmodule:: aidas.triggering

This module handles the generation of scan strategies and triggering of the lidar based on radar data.

.. autosummary::
    :toctree: generated/

    make_scan_file
    send_scan
    trigger_lidar_ppis_from_mask
    trigger_lidar_rhi_from_mask
"""
from .halo_lidar import make_scan_file, send_scan, trigger_lidar_ppis_from_mask, trigger_lidar_rhi_from_mask         # noqa