"""This module contains all the yolov5 models"""
import logging

import torch
from yolov5.models.common import AutoShape
from config.config import Config

def initialize_yolov5_model(config: Config = Config()) -> AutoShape:
    """Initializes the yolov5 model"""

    logging.getLogger("yolov5").setLevel(logging.WARNING)
    print(config)
    model = torch.hub.load(
        repo_or_dir=config.YOLOV5_REPO,
        model=config.CUSTOM_MODEL,
        path=config.BEST_MODEL_PATH,
        verbose=config.VERBOSE,
        force_reload=config.FORCE_RELOAD,
        device=config.DEVICE,
    )

    model.conf = config.CONF
    model.iou = config.IOU
    model.agnostic = config.AGNOSTIC
    model.multi_label = config.MULTI_LABEL
    model.classes = config.CLASSES
    model.max_det = config.MAX_DET
    model.amp = config.AMP
    return model


yolov5_model = initialize_yolov5_model()
# yolov5_table_detection_model = initialize_yolov5_model(TableDetectionConfig())
# yolov5_layout_detection_model = initialize_yolov5_model(LayoutDetectionConfig())
