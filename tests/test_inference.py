#!/usr/bin/env python
import pytest
import aidas
import torch
import numpy as np
import tempfile 
import pyart

"""Tests for `aidas` package."""

def test_infer_fcn_resnet50():
    torch.manual_seed(42)
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
    rad_scan = aidas.model.infer_lake_breeze(
        rad_scan, model_name='lakebreeze_best_model_fcn_resnet50')
    np.testing.assert_almost_equal(rad_scan.lakebreeze_mask.sum(), 700, decimal=-2)
    assert rad_scan.lakebreeze_mask.shape == (256, 256)

def test_infer_fcn_resnet50_no_augmentation():
    torch.manual_seed(42)
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
    rad_scan = aidas.model.infer_lake_breeze(rad_scan, model_name='lakebreeze_model_fcn_resnet50_no_augmentation')
    np.testing.assert_almost_equal(rad_scan.lakebreeze_mask.sum(), 1158, decimal=-2)
    assert rad_scan.lakebreeze_mask.shape == (256, 256)

def test_infer_fcn_resnet50_batch():
    torch.manual_seed(42)
    rad_scan1 = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
    rad_scan2 = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:05:00')
    rad_scan = aidas.model.infer_lake_breeze_batch(
        [rad_scan1, rad_scan2], model_name='lakebreeze_best_model_fcn_resnet50')
    assert len(rad_scan) == 2
    np.testing.assert_almost_equal(rad_scan[0].lakebreeze_mask.sum(), 766, decimal=-2)
    np.testing.assert_almost_equal(rad_scan[1].lakebreeze_mask.sum(), 770, decimal=-2)
    rad_scan = aidas.model.infer_lake_breeze_batch(
        [rad_scan1, rad_scan2], model_name='lakebreeze_model_fcn_resnet50_no_augmentation')
    assert len(rad_scan) == 2
    np.testing.assert_almost_equal(rad_scan[0].lakebreeze_mask.sum(), 1191, decimal=-2)
    np.testing.assert_almost_equal(rad_scan[1].lakebreeze_mask.sum(), 737, decimal=-2)


    # Test parallel processing
    with tempfile.TemporaryDirectory() as tmpdir:
        pyart.io.write_cfradial(
            f"{tmpdir}/radar1.nc", rad_scan1.pyart_object)
        pyart.io.write_cfradial(
            f"{tmpdir}/radar2.nc", rad_scan2.pyart_object)
        rad_scan = aidas.io.preprocess_radar_image_batch(
            f"{tmpdir}/*.nc", parallel=True)
        rad_scan = aidas.model.infer_lake_breeze_batch(
            rad_scan, model_name='lakebreeze_best_model_fcn_resnet50')
        assert len(rad_scan[:]) == 2
        np.testing.assert_almost_equal(rad_scan[0].sum(), 761, decimal=-2)
        np.testing.assert_almost_equal(rad_scan[1].sum(), 783, decimal=-2)
        np.testing.assert_almost_equal(rad_scan[0:2].sum(), 1544, decimal=-2)


def test_infer_fcn_resnet50_invalid_model():
    torch.manual_seed(42)
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
    with pytest.raises(ValueError):
        aidas.model.infer_lake_breeze(rad_scan, model_name='invalid_model_name')