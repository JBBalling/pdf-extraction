import pdfplumber
import os 
import numpy as np
import cv2 
import re
import logging

from image_utils import boxOverlap, transform_coordinate_IMG2PDF, box_inside_box, xyxy_xywh, bounding_box_refinement, xywh_xyxy
from extraction_utils import common_error_replacements, get_height_frequency, group_words, has_digit, is_digit, mark_footnote_start, mark_footnote_end, mark_superscript, get_bottom_coordinate_from_line
from yolov5_load import detect, get_results, load_model
from config.config import Config
from pdf2image import convert_from_path
from PIL import Image
from typing import Dict, List
from PIL.Image import Image as  ImageType
import time
import math

start_time = time.time()
pages = 0
pdf_path = "pdf-data"
output_path = "pdfplumber-output-xtolerance-1.5-doclaynet-with-mergeing"
pattern = re.compile(r"(.*verzeichnis(\n|$)|.*übersicht(\n|$)|.*\.\.\.\.|.*\. \. \. \.)")

if not os.path.exists(output_path):
    os.mkdir(output_path)

# only keep relevant classes ->
# caption (0), footnote (1), formula (2),
# list-item (3), page-footer (4), page header (5),
# picture (6), section-header (7), table(8), text (9), title (10)

CLASS_PRIORITIES = [10, 7, 3, 9]
DPI = 300
MARGIN = 0
DEVICE = "cpu"
HEADER_SECTION_MAX_TITLE_HEIGHT_RATIO = 0.02
FONT_MARGIN = 0.05
X_TOLERANCE = 1.5 # 1.5 for 200 DPI
Y_TOLERANCE = 3.5 # 3.5 for 200 DPI
START_PAGE = 0 # set to None for whole document
END_PAGE = 50 # set to None for whole document
SUPERSCRIPT_MARKER = "$$$"
LOG_FILENAME = "debug.log"


total_pages = 0
if os.path.exists(LOG_FILENAME):
    os.remove(LOG_FILENAME)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)

def visualize_page_layout(detection_list: List[Dict], images: List[ImageType], index: int, save_path: str) -> None:
    for i, (image, detections) in enumerate(zip(images, detection_list)):
        if i == index:
            numpy_image = np.array(image)

            def get_color_map(categories: List) -> Dict:
                color_map = {}
                distinct_categories = set(categories)
                colors = [np.random.randint(0, 255, 3) for cat in distinct_categories]
                for category, color in zip(distinct_categories, colors):
                    color_map[category] = (int(color[0]), int(color[1]), int(color[2]))

                return color_map

            color_map = get_color_map(detections["class"])
            for j in range(len(detections["xmin"])):
                xmin, ymin, xmax, ymax = (
                    int(detections["xmin"][j]) - MARGIN,
                    int(detections["ymin"][j]) - MARGIN,
                    int(detections["xmax"][j]) + MARGIN,
                    int(detections["ymax"][j]) + MARGIN,
                )
                txt = detections["name"][j]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]

                numpy_image = cv2.rectangle(
                    numpy_image,
                    (xmin - 5 * len(txt), ymin - cat_size[1] - 5),
                    (xmin - 5 * len(txt) + cat_size[0], ymin - 5),
                    color_map[detections["class"][j]],
                    -1,
                )

                numpy_image = cv2.putText(
                    numpy_image,
                    txt,
                    (xmin - 5 * len(txt), ymin - 5),
                    font,
                    0.5,
                    (255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                numpy_image = cv2.rectangle(numpy_image, (xmin, ymin), (xmax, ymax), color_map[detections["class"][j]], 2)

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            #Image.fromarray(numpy_image).show()
            Image.fromarray(numpy_image).save(f"{save_path}/image_{index+1}.png")


def deduplicate_detections(detections: List, images: List) -> List: 
    filtered_detections = []   
    for page in range(len(detections)):
        idx_to_remove = []
        i = 0
        while i < len(detections[page]["xmin"]): 
            bounding_box_i = [detections[page]["xmin"][i], detections[page]["ymin"][i], detections[page]["xmax"][i], detections[page]["ymax"][i]]
            class_i = detections[page]["class"][i]
            j = 0
            while j < len(detections[page]["xmin"]): 
                if i == j:
                    j += 1
                    continue
                
                bounding_box_j = [detections[page]["xmin"][j], detections[page]["ymin"][j], detections[page]["xmax"][j], detections[page]["ymax"][j]]
                class_j = detections[page]["class"][j]
                if boxOverlap(bounding_box_i, bounding_box_j) or box_inside_box(bounding_box_i, bounding_box_j):
                    # intersect
                    if CLASS_PRIORITIES.index(class_i) < CLASS_PRIORITIES.index(class_j) and i not in idx_to_remove:
                        idx_to_remove.append(i)
                    elif CLASS_PRIORITIES.index(class_i) == CLASS_PRIORITIES.index(class_j):
                        # try merging the boxes
                        merged_x1 = min(bounding_box_i[0], bounding_box_j[0])
                        merged_y1 = min(bounding_box_i[1], bounding_box_j[1])
                        merged_x2 = max(bounding_box_i[2], bounding_box_j[2])
                        merged_y2 = max(bounding_box_i[3], bounding_box_j[3])
                        detections[page]["xmin"][i] = merged_x1
                        detections[page]["ymin"][i] = merged_y1
                        detections[page]["xmax"][i] = merged_x2
                        detections[page]["ymax"][i] = merged_y2
                        # remove detection j from detections and indexes to remove and restart
                        for c in ["xmin", "ymin", "xmax", "ymax", "confidence", "name", "class"]:
                            detections[page][c].pop(j)
                        if j in idx_to_remove:
                            idx_to_remove.remove(j)
                        j = len(detections[page]["xmax"])
                        i = -1

                j += 1
            i += 1    
                
        for index in reversed(idx_to_remove):
            for c in ["xmin", "ymin", "xmax", "ymax", "confidence", "name", "class"]:
                detections[page][c].pop(index)
        filtered_detections.append(detections[page])


    return filtered_detections



def get_pdf_fontsize_statistic(pdf_doc, num_pages) -> Dict:
    n_pages = num_pages // 2
    middle_page = (len(pdf_doc.pages) // 2) // 2
    fontsize_statistics = {}
    for i in range(middle_page-n_pages, middle_page + n_pages):
        pdf_page = pdf_doc.pages[i]
        words_of_page = pdf_page.extract_words(x_tolerance=X_TOLERANCE, y_tolerance=Y_TOLERANCE)
        height_frequencies = get_height_frequency(words_of_page)
        for k, v in height_frequencies:
            k = round(k, 2)
            if k in fontsize_statistics:
                fontsize_statistics[k] += v
            else:
                fontsize_statistics[k] = v
    return sorted(fontsize_statistics.items(), key=lambda x: x[1], reverse=True)


def process_pdf(filepath: str, images: List) -> str:
    fulltext = ""
    # category = [3, 7, 9, 10]
    pdf_doc = pdfplumber.open(filepath)
    pdf_fontsize_statistics = get_pdf_fontsize_statistic(pdf_doc, 20)
    footnote_text_diff = pdf_fontsize_statistics[0][0] - pdf_fontsize_statistics[1][0]
    layout_detector = load_model(Config(), True, False, True, DEVICE, Config.CONF, CLASS_PRIORITIES, Config.IOU)
    detections = get_results(layout_detector, images)
    filtered_detections = deduplicate_detections(detections, images)
    pages = 0
    for index, detection in enumerate(filtered_detections):
        pdf_page = pdf_doc.pages[index]
        numpy_image = np.array(images[index])
        text = pdf_page.extract_text(x_tolerance=X_TOLERANCE, y_tolerance=Y_TOLERANCE)
        if pattern.search(text) or text == "":
            continue

        #words_of_page = pdf_page.extract_words(x_tolerance=X_TOLERANCE, y_tolerance=Y_TOLERANCE)
        #average_height = sum([w["height"] for w in words_of_page])/len(words_of_page)
        #grouped_by_size = group_words(words_of_page, 4, "height")
        pages += 1
        
        
        for (xmin, ymin, xmax, ymax, confidence, category, name) in sorted(zip(detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"], detection["confidence"], detection["class"], detection["name"]), key=lambda x: x[1]):
            crop_box = [
                int(xmin - MARGIN),
                int(ymin - MARGIN),
                int(xmax + MARGIN),
                int(ymax + MARGIN),
            ]
            crop_box_xywh = xyxy_xywh(crop_box)
            rect, bbox = bounding_box_refinement(numpy_image, None, crop_box_xywh, True)
            crop_box_extended = xywh_xyxy(bbox)
            pdf_coordinates = [transform_coordinate_IMG2PDF(c, DPI) for c in crop_box_extended]
            pdf_coordinates = [pdf_coordinates[0], pdf_coordinates[1], pdf_coordinates[2], pdf_coordinates[3]]
            cropped_page = pdf_page.crop(pdf_coordinates)
            text_from_box = cropped_page.extract_text(x_tolerance=X_TOLERANCE, y_tolerance=Y_TOLERANCE)
            if crop_box_xywh[3] <= (HEADER_SECTION_MAX_TITLE_HEIGHT_RATIO * numpy_image.shape[0]) and (crop_box_xywh[1] + crop_box_xywh[3]) <= 0.10 * numpy_image.shape[0]:
                LOGGER.info(f"Skipping Textbox with text:\n{text_from_box}\nbecause its in the headersection on page {index+1}")
                continue

            if (crop_box_xywh[1] + crop_box_xywh[3]) >= 0.80 * numpy_image.shape[0] and is_digit(text_from_box):
                LOGGER.info(f"Skipping Textbox with text:\n{text_from_box}\nbecause its in the page footer section on page {index+1}")
                continue

            words_from_box = cropped_page.extract_words(x_tolerance=X_TOLERANCE, y_tolerance=Y_TOLERANCE)
            if len(words_from_box) <= 0:
                continue
            average_height = sum([w["height"] for w in words_from_box])/len(words_from_box)
            lines = group_words(words_from_box, margin=(0.7*average_height))
            for line_index, line in enumerate(lines):
                footnote = False
                bottom_line_coordinate = round(get_bottom_coordinate_from_line(line), 2)
                common_line_height = get_height_frequency(line)[0][0]
                for word in line:
                    if round(word["height"], 2) > common_line_height:
                        # superscript
                        if has_digit(word["text"]):
                            word["text"] = mark_superscript(word["text"], SUPERSCRIPT_MARKER)
                        else:
                            continue
                    elif word["height"] < common_line_height and word["bottom"] > bottom_line_coordinate: 
                        # index
                        continue
                    elif (word["height"] < common_line_height and word["bottom"] < bottom_line_coordinate and has_digit(word["text"])) or (word["text"] == "¹") or (word["text"] == "²") or (word["text"] == "³"):
                        footnote = True
                        
                if footnote:
                    line = mark_footnote_start(line)
            
            text_block = ""
            for line in lines:
                text_block += " ".join(word["text"] for word in line) + "\n"
                
            text_block = common_error_replacements(text_block)
            fulltext += f"{text_block}\n\n"
            print(text_block)

        visualize_page_layout(filtered_detections, images, index, os.path.join(output_path, pdf_file.replace(".pdf", "")))
    return fulltext, pages


for pdf_file in os.listdir(pdf_path):
    # if pdf_file != "Dissertation_Dejnega.pdf":
    #     continue
    LOGGER.info(f"processing pdf file {pdf_file}")
    filtered_text = ""  
    filepath = os.path.join(pdf_path, pdf_file)
    images = convert_from_path(filepath, DPI, first_page=START_PAGE, last_page=END_PAGE)
    try:
        fulltext, processed_pages = process_pdf(filepath, images)
        total_pages += processed_pages
        with open(os.path.join(output_path, pdf_file.replace(".pdf", ".txt")), "w") as f:
            f.write(fulltext)
    except Exception as e:
        LOGGER.exception(e.with_traceback())    

end_time = time.time()
LOGGER.info(f"Processing took {end_time-start_time} seconds, for {len(os.listdir(pdf_path))} PDF and {total_pages} Pages.")