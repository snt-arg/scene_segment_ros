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
