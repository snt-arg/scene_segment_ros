import os
import gc
import torch
import torchvision
import numpy as np
from time import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def getFilteredSegments(predictions: dict, classes: list):
    """
    Returns a list of segments filtered by the defined classes

    Parameters
    -------
    predictions: dict
        The results generated by the model
    classes: list
        The list of classes to be filtered

    Returns
    -------
    filteredSegments: list
        The list of filtered segments
    filteredProbs: np.ndarray
        The matrix of per-pixel probabilities of the shape (H, W, C) 
    """
    # Initialize
    newSegmentInfo = []
    filteredSegments = predictions
    
    # Get the segment values and segment info
    segmentValues, segmentsInfo = predictions["panoptic_seg"]
    
    # flatten the class list
    classes_flat = []
    for c in classes:
        if isinstance(c, list):
            classes_flat.extend(c)
        else:
            classes_flat.append(c)

    # Iterate over the segmentInfo to find the desired segments
    for segment in segmentsInfo:
        # Filter the segments
        if segment["category_id"] in classes_flat:
            newSegmentInfo.append(segment)

    # Make a tuple
    filteredSegments["panoptic_seg"] = (segmentValues, newSegmentInfo)
    
    # Get probabilities and filter them
    # if there are lists in the classes list, need to add those logits together
    filteredProbs = torch.zeros(size=(filteredSegments["sem_seg"].shape[1], filteredSegments["sem_seg"].shape[2], len(classes)))
    permutedProbs = torch.permute(filteredSegments["sem_seg"], (1, 2, 0))
    for i, c in enumerate(classes):
        if isinstance(c, list):
            filteredProbs[:, :, i] = torch.sum(permutedProbs.index_select(-1, torch.tensor(c).to(device)), dim=-1).squeeze(-1)
        else:
            filteredProbs[:, :, i] = permutedProbs.index_select(-1, torch.tensor([c]).to(device)).squeeze(-1)

    filteredProbs = filteredProbs.cpu().detach().numpy()

    return filteredSegments, filteredProbs