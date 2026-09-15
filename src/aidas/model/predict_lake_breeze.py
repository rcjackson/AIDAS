import torch
import numpy as np

from huggingface_hub import hf_hub_download
from torch.nn import Identity 
from torchvision.models.segmentation import fcn_resnet50
from safetensors.torch import load_model
from scipy.ndimage import label

from ..io import RadarImage

# Identity layer needed for the lake-breeze detector model
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def infer_lake_breeze(radar_scan: RadarImage,
                    model_name='lakebreeze_model_fcn_resnet50_no_augmentation',
                    device='cpu',
                    area_threshold=20):
    """
    This module will infer the location of the lake breeze from a radar image.

    Parameters
    ----------
    radar_scan: :py:meth:`RadarImage`
        The RadarImage object containing the radar image.
    model_name: str
        The model to use. Currently, AIDAS has 2 models:
        'lakebreeze_model_fcn_resnet50_no_augmentation': 
        A fine-tuned ResNet50 with no data augmentation. Typically more liberal
        in detecting lake breezes.
        'lakebreeze_best_model_fcn_resnet50':
        A fine-tuned ResNet50 trained with data augmentation. Typically more conservative
        in detecting lake breezes.
    device: str
        The device to run the model on. Default is 'cpu'. Use 'cuda' for GPU inference.
    area_threshold: int
        The minimum continuous area in pixels for lake breeze segments. This helps
        remove false positive speckles that are identified by the model.

    Returns
    -------
    radar_scan: :py:meth:`RadarImage`
        The RadarImage with the lake breeze mask.
    """ 

    if model_name == 'lakebreeze_model_fcn_resnet50_no_augmentation':
        model = fcn_resnet50(num_classes=2)
    elif model_name == 'lakebreeze_best_model_fcn_resnet50':
        model = fcn_resnet50(weights='FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')
        in_channels = 2048
        inter_channels = 512
        channels = 2
        fcn_new_last_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(inter_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(inter_channels, channels, 1),
        )
        model.classifier = fcn_new_last_layer
        model.aux_classifier = Identity()
    else:
        raise ValueError(f"{model_name} is not a valid model.")

    state_dict = hf_hub_download(repo_id="rcjackson/lakebreeze-resnet50",
            filename=f"{model_name}.safetensors")
    load_model(model, state_dict)
    image = radar_scan.pytorch_image.to(device)
    model = model.to(device)
    mask = model(image)['out'].detach().numpy()
    mask = mask[0].argmax(axis=0)
    mask = mask.T

    # Filter out small regions that are not lake breezes
    labels, num_features = label(mask)
    largest_area = -99999
    for i in range(num_features):
        area = mask[labels == i+1].sum()
        if area > largest_area:
            largest_area = area
        if area < area_threshold:
            mask[labels == i+1] = 0
    radar_scan.lakebreeze_mask = mask
    return radar_scan

def infer_lake_breeze_batch(radar_list,
                            model_name='lakebreeze_best_model_fcn_resnet50',
                            area_threshold=20):
    """
    This module will infer the location of the lake breeze from a batch of radar images.

    Parameters
    ----------
    radar_scan: list of RadarImage
        The list of RadarImages containing the radar image.
    model_name: str
        The model to use. Currently, AIDAS has 2 models:
        *lakebreeze_model_fcn_resnet50_no_augmentation*: 
        A fine-tuned ResNet50 with no data augmentation. Typically more liberal
        in detecting lake breezes.
        *lakebreeze_best_model_fcn_resnet50*:
        A fine-tuned ResNet50 trained with data augmentation. Typically more conservative
        in detecting lake breezes.
    area_threshold: int
        The minimum continuous area in pixels for lake breeze segments. This helps
        remove false positive speckles that are identified by the model.

    Returns
    -------
    radar_scan: list of :py:meth:`RadarImage`
        The RadarImages with the lake breeze mask.
    """ 
    if model_name == 'lakebreeze_model_fcn_resnet50_no_augmentation':
        model = fcn_resnet50(num_classes=2)
    elif model_name == 'lakebreeze_best_model_fcn_resnet50':
        model = fcn_resnet50(weights='FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')
        in_channels = 2048
        inter_channels = 512
        channels = 2
        fcn_new_last_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(inter_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(inter_channels, channels, 1),
        )
        model.classifier = fcn_new_last_layer
        model.aux_classifier = Identity()
    else:
        raise ValueError(f"{model_name} is not a valid model.")
    state_dict = hf_hub_download(repo_id="rcjackson/lakebreeze-resnet50",
            filename=f"{model_name}.safetensors")
    load_model(model, state_dict)

    if isinstance(radar_list, list):
        image = torch.concat([x.pytorch_image for x in radar_list], axis=0)
    elif isinstance(radar_list, RadarImage):
        image = radar_list.pytorch_image

    mask = model(image)['out'].detach().numpy()
    mask = mask.argmax(axis=1)
    mask = np.transpose(mask, [0, 2, 1])

    # Filter out small regions that are not lake breezes
    if isinstance(radar_list, list):
        nsamples = len(radar_list)
    else:
        nsamples = mask.shape[0]
    for i in range(nsamples):
        labels, num_features = label(mask)
        largest_area = -99999
        for j in range(num_features):
            area = mask[labels == j+1].sum()
            if area > largest_area:
                largest_area = area
            if area < area_threshold:
                mask[labels == j+1] = 0
        if isinstance(radar_list, list):
            radar_list[i].lakebreeze_mask = mask[i]
    if isinstance(radar_list, RadarImage):
        radar_list.lakebreeze_mask = mask
    return radar_list
