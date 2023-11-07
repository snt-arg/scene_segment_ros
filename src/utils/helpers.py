import os
import gc
import torch
import torchvision


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
