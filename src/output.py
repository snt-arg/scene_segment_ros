import cv2
import torch
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from fastsam.utils import convert_box_xywh_to_xyxy
from utils.semantic_utils import label2rgb, ADE20K_COLOR_MAP


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


def FCNVisualizer(image, predictions, cfg):
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


def FCNEntropyVisualizer(image, predictions, cfg):
    """
    Shows the output of panoptic or instance segmentation
    with the entropies to show the confidence of the model

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
    semantic_segmentation = predictions["sem_seg"]
    # Compute the entropy
    entropy = -torch.sum(semantic_segmentation *
                         torch.log(semantic_segmentation+1e-10), axis=0) * 255
    entropy = entropy.to("cpu").numpy().astype(np.uint8)
    # Generate color map from probabilities (black 0 - white 1)
    color_map = cv2.applyColorMap(
        (entropy).astype(np.uint8), cv2.COLORMAP_BONE)

    return color_map


def SegFormerVisualizer(image, predictions):
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
    color_map = label2rgb(predictions_hot, image, ADE20K_COLOR_MAP)

    # Return
    return color_map


def SegFormerEntropyVisualizer(image, predictions):
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
    color_map = cv2.applyColorMap(
        (entropy).astype(np.uint8), cv2.COLORMAP_BONE)

    return color_map
