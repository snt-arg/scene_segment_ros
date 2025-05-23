import torch
import multiprocessing as mp
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from utils.helpers import getRootAbsolutePath, getFilteredSegments

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def fastSamInit(name: str, path: str):
    """
    Initializes Fast SAM (Semantic Anything Model) and returns the registered model

    Parameters
    -------
    name: str
        The name of the model (in this case, FAST-SAM)
    path: str
        The path to the model

    Returns
    -------
    model: str
        The initialized model
    """
    # Import
    from fastsam import FastSAM
    # Initialization
    print(f'Initializing "{name}" model ...')
    # Initialization
    model = FastSAM(path)
    print('Model loaded and is ready to use!\n')
    return model


def fastSamSegmenter(image, model, imageSize=640, conf=0.4, iou=0.9):
    """
    Segments the given image using Fast SAM

    Parameters
    -------
    image: Mat
        The input image for segmentation
    model: dict
        A registered model of Fast SAM
    imageSize: int
        The width of the input image (default: 640)
    conf: float
        The object confidence threshold (default: 0.4)
    iou: float
        Annotations filtering threshold (default: 0.4)

    Returns
    -------
    predictions: dict
        The predictions created by the model containing segments
    """
    # Import
    from fastsam import FastSAMPrompt
    # Generating mask (everything result)
    maskGenerator = model(
        image,
        device=DEVICE,
        retina_masks=True,
        imgsz=imageSize,
        conf=conf,
        iou=iou
    )
    # Process
    predictions = FastSAMPrompt(image, maskGenerator, device=DEVICE)
    return predictions


def pFCNInit(name: str, modelPath: str, configPath: str):
    """
    Initializes Detectron2 model based on Panoptic FCNconfig

    Parameters
    -------
    name: str
        The name of the model (in this case, FAST-SAM)
    modelPath: str
        The path to the model
    configPath: str
        The path to the model's specific configurations

    Returns
    -------
    model: str
        The initialized model
    """
    # Import
    from panopticfcn import add_panopticfcn_config
    # Initialization
    print(f'Initializing "{name}" model ...')
    # Convert to absolute path
    configPath = getRootAbsolutePath(configPath)
    modelPath = getRootAbsolutePath(modelPath)
    # Initialization
    cfg = get_cfg()
    add_panopticfcn_config(cfg)
    cfg.merge_from_file(configPath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = modelPath
    cfg.DEVICE = DEVICE.type
    model = DefaultPredictor(cfg)
    print('Model loaded and is ready to use!\n')
    return model, cfg


def pFCNSegmenter(image, model, classes):
    """
    Segments the given image using Panoptic FCN

    Parameters
    -------
    image: Mat
        The input image for segmentation
    model: DefaultPredictor
        A predictor model based on Detectron 2
    classes: list
        The list of classes to be filtered

    Returns
    -------
    predictions: dict
        The results generated by the model containing segments
    """
    predictions = model(image)
    filteredPreds, predictionProbs = getFilteredSegments(predictions, classes)
    return filteredPreds, predictionProbs


def SegFormerInit(name: str):
    """
    Initializes Transformer model based on SegFormer
    Source: https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation

    Parameters
    -------
    name: str
        The name of the model (in this case, SegFormer)
    modelPath: str
        The path to the model
    configPath: str
        The path to the model's specific configurations

    Returns
    -------
    model: str
        The initialized model
    """
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    print(f'Initializing "{name}" model ...')
    # Convert to absolute path
    image_processor = AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512")
    model.eval()
    print('Model loaded and is ready to use!\n')
    return model, image_processor


def SegFormerSegmenter(image, model: SegformerForSemanticSegmentation, image_processor: AutoImageProcessor):
    """
    Segments the given image using Panoptic FCN

    Parameters
    -------
    image: Mat
        The input image for segmentation
    model: DefaultPredictor
        A predictor model based on Detectron 2
    classes: list
        The list of classes to be filtered

    Returns
    -------
    predictions: dict
        The results generated by the model containing segments
    """
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # shape (batch_size, num_labels, height/4, width/4)
    logits = outputs.logits
    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    # Filter the segments
    # Return
    return probs


def maskDinoInit(name: str, modelPath: str, configPath: str, confidence=0.5):
    """
    Initializes Detectron2 model based on Panoptic FCNconfig

    Parameters
    -------
    name: str
        The name of the model (in this case, FAST-SAM)
    modelPath: str
        The path to the model
    configPath: str
        The path to the model's specific configurations

    Returns
    -------
    model: str
        The initialized model
    """
    # Import
    from maskdino import add_maskdino_config, VisualizationDemo
    # Initialization
    print(f'Initializing "{name}" model ...')
    mp.set_start_method("spawn", force=True)
    # Convert to absolute path
    configPath = getRootAbsolutePath(configPath)
    modelPath = getRootAbsolutePath(modelPath)
    # Configs
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(configPath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.WEIGHTS = modelPath
    cfg.MODEL.DEVICE = DEVICE.type
    cfg.freeze()
    # Create the model
    model = VisualizationDemo(cfg)
    print('Model loaded and is ready to use!\n')
    return model


def maskDinoSegmenter(image, model, classes):
    """
    Segments the given image using Panoptic FCN

    Parameters
    -------
    image: Mat
        The input image for segmentation
    model: DefaultPredictor
        A predictor model based on Detectron 2
    classes: list
        The list of classes to be filtered

    Returns
    -------
    predictions: dict
        The results generated by the model containing segments
    """
    predictions = model.segment(image)
    return predictions


def yosoInit(name: str, modelPath: str, configPath: str, confidence=0.5, overlap=0.98):
    """
    Initializes Detectron2 DEfaultPredictor model based 
    on YOSO config and returns it,

    Returns
    -------
    model: DefaultPredictor
        A detector model
    cfg: cfgNode
        The configuration to be later used in visualization
    """
    # Import
    from yoso.utils import addYosoConfig
    from yoso.yoso.segmentator import YOSO
    # Initialization
    print(f'Initializing "{name}" model ...')
    # Convert to absolute path
    configPath = getRootAbsolutePath(configPath)
    modelPath = getRootAbsolutePath(modelPath)
    # Configs
    cfg = get_cfg()
    addYosoConfig(cfg)
    cfg.merge_from_file(configPath)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence
    cfg.MODEL.YOSO.TEST.OVERLAP_THRESHOLD = overlap
    cfg.MODEL.YOSO.TEST.OBJECT_MASK_THRESHOLD = confidence
    cfg.MODEL.WEIGHTS = modelPath
    cfg.MODEL.DEVICE = DEVICE.type
    cfg.freeze()
    # Create the model
    model = DefaultPredictor(cfg)
    print('Model loaded and is ready to use!\n')
    return model, cfg


def yosoSegmenter(image, model, classes):
    """
    Segments the given image using Panoptic FCN

    Parameters
    -------
    image: Mat
        The input image for segmentation
    model: DefaultPredictor
        A predictor model based on Detectron 2
    classes: list
        The list of classes to be filtered

    Returns
    -------
    filteredSegments: dict
        The dictionary of filtered segments
    filteredProbs: np.ndarray
        The matrix of per pixel probabilities of shape (W, H, C)
    """
    predictions = model(image)
    filteredSegments, filteredProbs = getFilteredSegments(predictions, classes)
    return filteredSegments, filteredProbs
