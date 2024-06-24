import os
import gc
import torch
import torchvision
import numpy as np


def monitorParams():
    """
    Prints a summary of various parameters
    """
    print("\nChecking general status ...")
    print(" * PyTorch version:", torch.__version__)
    print(" * Torchvision version:", torchvision.__version__)
    print(" * CUDA is available:", torch.cuda.is_available())
    print()
    # print(torch.cuda.memory_summary(
    #     device=None, abbreviated=True))


def cleanMemory():
    """
    Cleans memory on GPU
    """
    torch.cuda.empty_cache()
    gc.collect()


def getRootAbsolutePath(relativePath: str):
    """
    Returns the absolute path of the root directory

    Parameters
    -------
    relativePath: str
        The relative path to the root directory

    Returns
    -------
    absolutePath: str
        The absolute path of the root directory
    """
    # Get the directory of the current script
    scriptDirectory = os.path.dirname(__file__)
    # Get the root directory of the repository
    root = os.path.dirname(os.path.dirname(scriptDirectory))
    # Get the absolute path of the current directory
    absolutePath = os.path.abspath(
        os.path.join(root, relativePath))
    # Return
    return absolutePath


def getFCNFilteredSegments(segments: dict, classes: list):
    """
    Returns a list of segments filtered by the defined classes

    Parameters
    -------
    segments: list
        The list of segments
    classes: list
        The list of classes to be filtered

    Returns
    -------
    newSegments: list
        The list of filtered segments
    predictionProbs: list
        The list of probabilities of the filtered segments
    """
    # Initialize
    newSegmentInfo = []
    newSegments = segments
    # Get the segment values and segment info
    segmentValues, segmentsInfo = segments["panoptic_seg"]
    # Iterate over the segmentInfo to find the desired segments
    for segment in segmentsInfo:
        # Filter the segments
        if segment["category_id"] in classes:
            newSegmentInfo.append(segment)
    # Make a tuple
    newSegments["panoptic_seg"] = (segmentValues, newSegmentInfo)
    # Get probabilities
    predictionProbs = torch.permute(
        newSegments["sem_seg"], (1, 2, 0)).to("cpu").numpy()
    # Take only the probabilities for classes needed from params
    predictionProbs = np.take(predictionProbs, classes, -1)
    # Return
    return newSegments, predictionProbs


def getYosoFilteredSegments(segments: dict, classes: list):
    """
    Returns a list of segments filtered by the defined classes

    Parameters
    -------
    segments: list
        The list of segments
    classes: list
        The list of classes to be filtered

    Returns
    -------
    filteredSegments: list
        The list of filtered segments
    """
    # Initialize
    newSegmentInfo = []
    newSegments = segments
    # Get the segment values and segment info
    segmentValues, segment_dict, segmentsInfo = segments["panoptic_seg"]
    # Iterate over the segmentInfo to find the desired segments
    for segment in segmentsInfo:
        # Filter the segments
        if segment["category_id"] in classes:
            newSegmentInfo.append(segment)
    # Make a tuple
    newSegments["panoptic_seg"] = (segment_dict, newSegmentInfo)
    # Get probabilities
    predictionProbs = torch.permute(
        segmentValues, (2, 1, 0)).to("cpu").numpy()
    # Take only the probabilities for classes needed from params
    predictionProbs = np.take(predictionProbs, classes, -1)
    # Return
    return newSegments, segment_dict, predictionProbs
