import cv2
import torch
import numpy as np
from detectron2.data import MetadataCatalog
from fastsam.utils import convert_box_xywh_to_xyxy
from utils.semantic_utils import label2rgb, ADE20K_COLOR_MAP
from detectron2.utils.visualizer import Visualizer, ColorMode


def fastSamVisualizer(masks, pointPrompt, boxPrompts, pointLabel, counters):
    """
    Shows the output segmented image

    Parameters
    -------
    masks: Any
        List of masks provided by the segmenter

    Returns
    -------
    result: Mat
        The segmented visualized image
    """
    # Init
    bboxes = None
    points = None
    textPrompt = None
    # Read values
    boxPrompt = convert_box_xywh_to_xyxy(boxPrompts)
    # Annotations
    if boxPrompt[0][2] != 0 and boxPrompt[0][3] != 0:
        ann = masks.box_prompt(bboxes=boxPrompt)
        bboxes = boxPrompt
    elif textPrompt != None:
        ann = masks.text_prompt(text=textPrompt)
    elif pointPrompt[0] != [0, 0]:
        ann = masks.point_prompt(
            points=pointPrompt, pointlabel=pointLabel
        )
        points = pointPrompt
    else:
        ann = masks.everything_prompt()
    # Plotting
    result = masks.plot(
        annotations=ann,
        output_path='',
        bboxes=bboxes,
        points=points,
        point_label=pointLabel,
        withContours=counters,
        better_quality=False,
    )
    # Return
    return result


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


def segFormerVisualizer(image, predictions):
    """
    Shows the output of panoptic or instance segmentation

    Parameters
    -------
    image: Mat
        The input image
    predictions: np.ndarray
        Matrix with class probabilities
    cfg: CfgNode
        The configuration object

    Returns
    -------
    result: Mat
        The segmented visualized image
    """
    # Init
    predictions = predictions.to("cpu")
    # Generate color map from predictions (labels)
    predictions_hot = predictions.argmax(dim=0).numpy()
    colorMap = label2rgb(predictions_hot, image, ADE20K_COLOR_MAP)
    # Return
    return colorMap


def segFormerEntropyVisualizer(predictions):
    """
    Shows the output of panoptic or instance segmentation

    Parameters
    -------
    image: Mat
        The input image
    predictions: np.ndarray
        Matrix with class probabilities

    Returns
    -------
    result: Mat
        The segmented visualized image
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


def maskDinoVisualizer(image, model, predictions):
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
    ranModel = model.run_on_image(image, predictions)
    # Extract the image
    colorMap = ranModel.get_image()[:, :, ::-1]
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
