"""This module contains all the detection related methods"""
import logging
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict
from PIL.Image import Image as ImageType
import cv2
import torch
from PIL import Image
from PIL.Image import Image as TypeImage
from yolov5.models.common import AutoShape, Detections

from config.config import Config
#from src.models import TypePath
#from src.models.type_hints import Table_Detection_Categories

from yolov5_model import yolov5_model




def get_relative_coordinates(bboxes: Dict, image: ImageType) -> Dict[int, List[float]]:
    """returns the relative coordinates of a bounding box with respect to its image

    Args:
        bboxes [Dict]: the dictionary containing the bounding boxes from objects on the page
        image [ImageType]: the respective PIL Image

    Returns:
        relative_coordinates [Dict[int, List[float]]]: a dictionary with the object index as key and its relative coordinates as List
    """
    relative_coordinates = {}
    if len(bboxes["xmin"]) > 0:
        for index in bboxes["xmin"].keys():
            relative_coordinates[index] = [
                bboxes["xmin"][index] / image.size[0],
                bboxes["ymin"][index] / image.size[1],
                bboxes["xmax"][index] / image.size[0],
                bboxes["ymax"][index] / image.size[1],
            ]
    return relative_coordinates

def load_model(
    config: Config = Config(),
    force_reload: bool = True,
    verbose: bool = True,
    use_initialized: bool = True,
    device: Literal["gpu", "cpu"] = "cpu",
    conf: float = 0.45,
    classes: List = None,
    iou: float = 0.45,
) -> AutoShape:
    """
    Loads the yolov5 model
    Args:
        config: configuration for the model
        force_reload: force reload the model
        verbose: verbose option
        use_initialized: if we want to use the initialized model
        device: cpu or gpu
        model_type: the yolov5 model_type to be used
        conf: the confidence to use for the predictions
        classes: the value(s) of the class id(s) to detect

    Returns:
        The specified yolov5 model
    """
    if use_initialized:
        model = yolov5_model
        model = update_model(model, conf=conf, classes=classes, iou=iou)
        return model
    else:
        if not verbose:
            logging.getLogger("yolov5").setLevel(logging.WARNING)

        model = torch.hub.load(
            repo_or_dir=Config.YOLOV5_REPO,
            model=Config.CUSTOM_MODEL,
            path=Config.BEST_MODEL_PATH,
            force_reload=force_reload,
            verbose=verbose,
        )
        model.conf = config.CONF
        model.iou = config.IOU
        model.agnostic = config.AGNOSTIC
        model.multi_label = config.MULTI_LABEL
        model.classes = config.CLASSES
        model.max_det = config.MAX_DET
        model.amp = config.AMP
        model.to(device)
        model = update_model(model, conf=conf, classes=classes, iou=iou)
    return model


def update_model(model: AutoShape, **kwargs) -> AutoShape:
    """
    Updates the model parameters, add more parameters when needed

    Args:
        model [AutoShape]: the model
        conf [float]: the prediction confidence

    Returns:
        model [AutoShape]: the updated model with the given parameters
    """
    model.__dict__.update(kwargs)
    return model


def load_image(image_path: Union[Path, str]) -> TypeImage:
    """
    Loads an image in PIL format
    Args:
        image_path: The path to the image

    Returns:
        The image
    """
    return Image.open(image_path)


def load_images(images_folder: Union[Path, str], fmt: str = "png") -> List[TypeImage]:
    """
    Loads all the images from in a folder in PIL format
    Args:
        images_folder: The folder containing the images
        fmt: The format of the images

    Returns:
        List of the images
    """
    image_files = sorted(Path(images_folder).glob(f"*.{fmt}"))
    images = []
    for image_path in image_files:
        images.append(load_image(image_path))
    return images


def save_detections(
    images_or_path: Union[Union[Path, str], List[TypeImage]],
    save_dir: Union[Path, str],
    model: AutoShape,
    size: Literal[640, 1280] = 640,
    fmt: str = "png",
) -> None:
    """
    Detects the figures on the images saves the output

    Args:
        images_or_path: The folder of the images or the list of the images
        size: The size of the image
        save_dir: The directory to save the figures
        model: the model
        fmt: format of the images
    """
    results = detect(images_or_path=images_or_path, model=model, size=size, fmt=fmt)
    return results.save(save_dir=save_dir)


def crop_images(
    images_or_path: Union[Union[Path, str], List[TypeImage]],
    model: AutoShape,
    size: Literal[640, 1280] = 640,
    save: bool = True,
    save_dir: Optional[Union[Path, str]] = None,
    fmt: str = "png",
) -> List[dict]:
    """
    Detects the figures on the images and crops the output

    Args:
        images_or_path: The folder of the images or the list of the images
        model: the model
        size: The size of the images
        save: If we want to save the cropped figures
        save_dir: The directory to save the cropped figures
        fmt: format of the images
    """
    results = detect(images_or_path=images_or_path, model=model, size=size, fmt=fmt)
    return results.crop(save=save, save_dir=save_dir)


def detect(
    images_or_path: Union[Union[Path, str], List[TypeImage]], model: AutoShape, size: Literal[640, 1280] = 640, fmt: str = "png"
) -> Detections:
    """
    Detects the figures on the images
    Args:
        images_or_path: The folder of the images or the list of the images
        model: the model
        size: The size of the images
        fmt: format of the images

    Returns:
        The detected objects
    """
    if not isinstance(images_or_path, list):
        images = load_images(images_or_path, fmt=fmt)
    else:
        images = images_or_path
    return model(images, size=size)


def detections_bytes(
    images_or_path: Union[Union[Path, str], List[TypeImage]],
    model: AutoShape,
    size: Literal[640, 1280] = 640,
    fmt: str = "png",
) -> List[bytes]:
    """
    Detects the figures and returns them as a list of bytes
    Args:
        images_or_path: The folder of the images or the list of the images
        model: the model
        size: The size of the images
        fmt: format of the images

    Returns:
        A list of bytes of the detected figures
    """
    results = detect(images_or_path=images_or_path, model=model, size=size, fmt=fmt)
    images_bytes = []
    for image in results.render():
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, encoded_image = cv2.imencode(".png", image)
        images_bytes.append(encoded_image.tobytes())
    return images_bytes


def get_results(model, images) -> List:
        """Gets the object detections of the pdf"""
        results = detect(images, model)


        detections = []
        for result, image in zip(results.pandas().xyxy, images):
            bb = {
                "xmin": [],
                "ymin": [],
                "xmax": [],
                "ymax": [],
                "confidence": [],
                "class": [],
                "name": [],
            }
            bboxes = result.to_dict()
            for key, values in bboxes.items():
                for _, val in values.items():
                    bb[key].append(val)
            
            
            detections.append(bb)
        return detections