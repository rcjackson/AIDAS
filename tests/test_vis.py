import pytest
import aidas
import torch
import numpy as np
import matplotlib.pyplot as plt

@pytest.mark.mpl_image_compare(tolerance=50)
def test_visualize_lake_breeze():
    torch.manual_seed(42)
    rad_scan = aidas.io.preprocess_radar_image('KLOT', '2025-07-15T18:00:00')
    rad_scan = aidas.model.infer_lake_breeze(
        rad_scan, model_name='lakebreeze_best_model_fcn_resnet50')
    
    fig, ax = aidas.vis.visualize_lake_breeze(rad_scan, bg_field='reflectivity')
    assert fig is not None
    assert ax is not None
    return fig 