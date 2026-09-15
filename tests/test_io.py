import pytest 
import aidas
import torch
import numpy as np


def test_preprocess_radar_image():
    torch.manual_seed(42)
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
    assert rad_scan.pytorch_image.shape == torch.Size([1, 3, 256, 256])
    assert rad_scan.pyart_object.fields['reflectivity']['data'].shape == (6480, 1832)
    np.testing.assert_almost_equal(rad_scan.pytorch_image.numpy().sum(), 150341140.0, decimal=-4)
