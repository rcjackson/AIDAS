"""
============================
aidas.testing (aidas.testing)
============================

.. currentmodule:: aidas.testing

This module handles testing utilities for AIDAS.


.. autosummary::
    :toctree: generated/

    FakeSSHClient
    FakeSFTP
    TEST_RHI_FILE
    TEST_PPI_FILE
    TEST_PPI_TRIGGERED_SCAN
    TEST_RHI_TRIGGERED_SCAN
"""
import os

from .fake_lidar import FakeSFTP, FakeSSHClient        # noqa

TEST_RHI_FILE = os.path.join(os.path.dirname(__file__), "data/test_scan_rhi.txt")
TEST_PPI_FILE = os.path.join(os.path.dirname(__file__), "data/test_scan_ppi.txt")
TEST_PPI_TRIGGERED_SCAN = os.path.join(os.path.dirname(__file__), "data/test_scan_ppi_lakebreeze_close.txt")
TEST_RHI_TRIGGERED_SCAN = os.path.join(os.path.dirname(__file__), "data/test_scan_rhi_lakebreeze.txt")
