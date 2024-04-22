import numpy as np
import cv2


from numpy import ndarray
from typing import Dict, Optional, Tuple, List



def bounding_box_refinement(
    img: Optional[ndarray] = None,
    image_path: Optional[str] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
    expansion: Optional[bool] = None,
) -> Tuple[ndarray, Tuple[int, int, int, int]]:
    """refines the bounding boxes boundaries to perfectly match the internal component

    Args:
        img [ndarray]: the image
        image_path [Optional[str]]: the image_path if not img
        box [Optional[Tuple[int, int, int, int]]]: the bounding box [x, y, w, h]
        expansion [Optional[bool]]: wether to expand bbox or not

    Returns:
        rect [ndarray]: the cropped part of the image
        bbox [Tuple[int, int, int, int]]: the expanded bounding box
    """
    if img is None:
        img = cv2.imread(image_path)


    x, y, w, h = box

    if expansion:
        x, y, w, h = bounding_box_expansion(x, y, w, h)

    bounding_box = img[y : y + h, x : x + w]

    x_edge, y_edge, w_edge, h_edge = edge_detection(bounding_box)

    rect = bounding_box[y_edge : y_edge + h_edge, x_edge : x_edge + w_edge]

    x_new = x + x_edge
    y_new = y + y_edge

    bbox = x_new, y_new, w_edge, h_edge
    return rect, bbox


def bounding_box_expansion(x: int, y: int, w: int, h: int, length: int = 4) -> Tuple[int, int, int, int]:
    """Expands the bounding box"""
    return x - length, y - length, w + 2 * length, h + 2 * length


def edge_detection(img: ndarray) -> Tuple[int, int, int, int]:
    """
    Detects the edges of an image and returns the bounding box by excluding the white spaces
    Follows this tutorial: https://learnopencv.com/edge-detection-using-opencv/
    Args:
        img: The image as a numpy.ndarray

    Returns:
        The x, y, w, h of the bounding box
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    coords = cv2.findNonZero(edges)
    return cv2.boundingRect(coords)


def xyxy_xywh(bbox: List) -> List:
    return [
        bbox[0],
        bbox[1], 
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
    ]

def xywh_xyxy(bbox: List) -> List:
    return [
        bbox[0],
        bbox[1], 
        bbox[2] + bbox[0],
        bbox[3] + bbox[1]
    ]


def transform_coordinate_IMG2PDF(coordinate: float, dpi: int) -> int:
    # pdf_dpi = 72,
    return int((coordinate * 72) / dpi)



def boxOverlap(box1, box2):
    # get corners
    tl1 = (box1[0], box1[1])
    br1 = (box1[2], box1[3])
    tl2 = (box2[0], box2[1])
    br2 = (box2[2], box2[3])

    # separating axis theorem
    # left/right
    if tl1[0] >= br2[0] or tl2[0] >= br1[0]:
        return False

    # top/down
    if tl1[1] >= br2[1] or tl2[1] >= br1[1]:
        return False

    # overlap
    return True


def box_inside_box(box1, box2):
    tl1 = (box1[0], box1[1])
    br1 = (box1[2], box1[3])
    tl2 = (box2[0], box2[1])
    br2 = (box2[2], box2[3])
    return (tl1[0] >= tl2[0] and tl1[1] >= tl2[1] and br1[0] <= br2[0] and br1[1] <= br2[1]) or \
        (tl2[0] >= tl1[0] and tl2[1] >= tl1[1] and br2[0] <= br1[0] and br2[1] <= br1[1])