import cv2
import torch
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


def pFCNVisualizer(image, predictions, cfg):
    """
    Shows the output of panoptic or instance segmentation

    Parameters
    -------
    image: Mat
        The input image
    predictions: dict
        Dict of segmentation results
    cfg: CfgNode
        The configuration object

    Returns
    -------
    result: Mat
        The segmented visualized image
    """
    # Init
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    visualizer = Visualizer(image[:, :, ::-1],
                            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    result = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to("cpu"), segments_info)
    # Return
    return result.get_image()[:, :, ::-1]


def entropyVisualizer(predictions):
    """
    Shows the output of panoptic or instance segmentation
    with the entropies to show the confidence of the model

    Parameters
    -------
    predictions: torch.Tensor
        Tensor with class probabilities of shape (C, H, W)

    Returns
    -------
    result: Mat
        The uncertainty image
    """
    # Compute the entropy
    entropy = -torch.sum(predictions *
                         torch.log(predictions+1e-10), axis=0) * 255
    entropy = entropy.to("cpu").numpy().astype(np.uint8)
    # Generate color map from probabilities (black 0 - white 1)
    colorMap = cv2.applyColorMap(
        (entropy).astype(np.uint8), cv2.COLORMAP_BONE)
    # Return
    return colorMap

def yosoVisualizer(image, predictions, cfg):
    """
    Shows the output of panoptic or instance segmentation

    Parameters
    -------
    image: Mat
        The original image that was to be segmented
    model: VisualizationDemo
        The demo model for MaskDino
    predictions: dict
        Dict of segmentation results
    """
    # Generate color map from predictions (labels)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    panopticSeg, segments_info = predictions["panoptic_seg"]
    ranModel = visualizer.draw_panoptic_seg_predictions(
        panopticSeg.to(torch.device("cpu")), segments_info)
    # Extract the image
    colorMap = ranModel.get_image()[:, :, ::-1]
    # Return
    return colorMap
