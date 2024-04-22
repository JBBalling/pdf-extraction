import os
import detectron2.data.transforms as T
import numpy as np
import cv2
import torch
import logging, logging.config
import time
import pdfplumber
import re 


from typing import Dict, List
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.data import MetadataCatalog

from pdf2image import convert_from_path
from PIL import Image
from PIL.Image import Image as Imagetype

cwd = os.getcwd()
#logging.config.fileConfig(os.path.join(cwd, "logging.conf"))
LOG = logging.getLogger(__name__)


class ObjectDetection():
    
    cpu = torch.device("cpu")
    def __init__(self, config_path: str, weights_path: str, labels: List, offset: int = 0, min_confidence: float = 0.5, device: str = "cuda") -> None:
        self.config_path = config_path
        self.weights_path = weights_path
        self.offset = offset
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.config_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_confidence  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = self.weights_path
        self.cfg.MODEL.DEVICE='cpu'
        self.model = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get('docbank_seg_test')
        # self.metadata.thing_classes = self.categories
        self.metadata.thing_classes = labels
        # print(t)
    

    def detect(self, image:np.ndarray) -> Dict:
        instances = {}
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        output = self.model(image)
        if "instances" in output:
            instances = output["instances"]
            
        return instances





def visualize_bboxes(image: Imagetype, detections: Dict, index: int, pdf_file: str) -> None:
    numpy_image = np.array(image)
    detections = detections.to(torch.device("cpu"))
    #print(detections.pred_boxes)

    vis = Visualizer(numpy_image)
    visualised_iamge = vis.draw_instance_predictions(detections)
    #print(type(visualised_iamge))
    img = visualised_iamge.get_image()
    #Image.fromarray(img).show()
    if not os.path.exists(pdf_file):
        os.mkdir(pdf_file)
    Image.fromarray(img).save(f"{pdf_file}/page_{index+1}.png")
  
labels = ["abstract", "author", "caption", "date", "equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"]
object_detector = ObjectDetection(
    config_path=os.path.join(cwd, "models", "docbank", "X101", "X101.yaml"),
    weights_path=os.path.join(cwd, "models", "docbank", "X101", "model.pth"),
    labels=labels,
    min_confidence=0.2
)
pattern = re.compile(r"(.*verzeichnis(\n|$)|.*übersicht(\n|$)|.*\.\.\.\.|.*\. \. \. \.)")


start_time = time.time()
pdf_path = "pdf-data"
output_path = "pdfplumber-output-xtolerance-1.5-docbank"
if not os.path.exists(output_path):
    os.mkdir(output_path)
    
pages = 0


for pdf_file in os.listdir(pdf_path):
    image_path = os.path.join(output_path, pdf_file.replace(".pdf", ""))
    print(f"processing pdf file {pdf_file}")
    filtered_text = ""  
    filepath = os.path.join(pdf_path, pdf_file)
    images = convert_from_path(filepath, first_page=0, last_page=40)
    pdf_doc = pdfplumber.open(filepath)
    for index, img in enumerate(images):
        # img.show()
        pdf_page = pdf_doc.pages[index]
        text = pdf_page.extract_text(x_tolerance=1.5)
        if pattern.search(text) or text == "":
            continue
        detections = object_detector.detect(img)
        # print(detections)
        visualize_bboxes(img, detections, index, os.path.join(output_path, pdf_file.replace(".pdf", "")))




