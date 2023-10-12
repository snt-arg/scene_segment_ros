import os
import torch
from fastsam import FastSAM, FastSAMPrompt

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def fastSamInit(name: str, path: str):
    """
    Initializes Fast SAM (Semantic Anything Model) and returns the registered model

    Returns
    -------
    model: str
        The name of the model (in this case, FAST SAM)
    path: str
        The path to the model (in this case, FAST SAM)
    """
    print(f'Initializing "{name}" model ...')
    # Initialization
    model = FastSAM(path)
    print('Model loaded and is ready to use!\n')
    return model


def fastSamSegmenter(image, model):
    """
    Segments the given image using Fast SAM

    Parameters
    -------
    image: Mat
        The input image for segmentation
    model: dict
        A registered model of Fast SAM

    Returns
    -------
    masks: dict
        The masks created by the model containing segments
    """
    print()
    # Generating mask (everything result)
    # maskGenerator = model(
    #     image,
    #     device=DEVICE,
    #     retina_masks=True,
    #     imgsz=fastSam_imageSize,
    #     conf=fastSam_conf,
    #     iou=fastSam_iou
    # )
    # # Process
    # masks = FastSAMPrompt(image, maskGenerator, device=DEVICE)
    # return masks
