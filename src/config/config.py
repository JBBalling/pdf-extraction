"""This module contains configuration parameters"""
import json
import os
from pathlib import Path
from typing import Dict, List, Literal




class Config:
    """The main configuration"""

    BASE_DIR: str = Path(__file__).parent.parent.parent
    DEBUG: bool = bool(os.getenv("DEBUG", False))
    APP_HOST: str = os.getenv("PDFINFOEXTRACTOR_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("PDFINFOEXTRACTOR_PORT", 5000))

    # API Variables
    OPEN_API_URL: str = "/openapi.json"
    API_PREFIX: str = ""
    DOCS_URL: str = "/docs"

    # Model Variables
    YOLOV5_REPO: str = "ultralytics/yolov5"
    CUSTOM_MODEL: str = "custom"
    BEST_MODEL_PATH: str = BASE_DIR / "models" / "yolov5m" / "layout_segmentation.pt"
    CONF: float = 0.2  # NMS confidence threshold
    IOU: float = 0.45  # NMS IoU threshold
    AGNOSTIC: bool = False  # NMS class-agnostic
    MULTI_LABEL: bool = False  # NMS multiple labels per box
    CLASSES: bool = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    MAX_DET: int = 20  # maximum number of detections per image
    AMP: bool = False  # Automatic Mixed Precision (AMP) inference
    DEVICE: Literal["cpu", "gpu"] = "cpu"
    VERBOSE: bool = False
    FORCE_RELOAD: bool = False

    DATA_PATH = BASE_DIR / "data" / "crops"
    TMP_PATH = BASE_DIR / "tmp"
