import pdfplumber
import os
import time
import re
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from typing import Tuple


pattern = re.compile(r"(.*verzeichnis(\n|$)|.*übersicht(\n|$)|.*\.\.\.\.|.*\. \. \. \.)")
start_time = time.time()
pdf_path = "pdf-data"
output_path = "pdfplumber-output-xtolerance-1.5-and-fixed-page-crop"
if not os.path.exists(output_path):
    os.mkdir(output_path)
    
pages = 0
FIXED_CROP_TOP = 0.10
FIXED_CROP_BOTTOM = 0.95

def group(l: list, group_range: int, sorting_index: int) -> list:
    """Groups vertical and horizontal lines by specific group range. This is needed to combine several lines into one line.
    
    Args:
        l (list): list of vertical or horizontal lines
        group_range (int): offset in positive and negative direction to group lines which meet in range
        sorting_index (int): 0 -> vertical lines, 1 -> horizontal lines

    Returns:
        list of lines grouped together by range
    """
    groups = []
    this_group = []
    i = 0
    l = list(l)
    l = sorted(l, key=lambda lines: lines[sorting_index])
    while i < len(l):
        a = l[i]
        if len(this_group) == 0:
            if i == len(l) - 1:
                # add last group
                this_group.append(a)
                break
            this_group_start = a
        if (
            a[sorting_index] <= this_group_start[sorting_index] + group_range
            and a[sorting_index] >= this_group_start[sorting_index] - group_range
        ):
            this_group.append(a)
        if a[sorting_index] < this_group_start[sorting_index] + group_range:
            i += 1
        else:
            groups.append(this_group)
            this_group = []
    groups.append(this_group)
    return [list(g) for g in groups if len(g) != 0]


def get_maximum_textline_length(group):
    max_length = 0
    for line in group:
        length = line[-1]["x1"] - line[0]["x0"]
        if length > max_length:
            max_length = length
    
    return max_length

def get_crop_height(pdf_doc) -> Tuple[int, int]:
    mid_of_document = len(pdf_doc.pages) // 2
    crop_from_top, crop_from_bottom = None, None
    page_width, page_height = 595, 842 # init with standard values for DINA4 PDF
    top_crops, bottom_crops = [], []
    for page in pdf_doc.pages[mid_of_document-3:mid_of_document+3]:
        page_width, page_height = page.width, page.height
        text = page.extract_words(x_tolerance=1.5)
        grouped_lines = group(text, 5, "top")
        max_textline_length = get_maximum_textline_length(grouped_lines)
        edges_greater_than_textline_length = []
        rects_greater_than_textline_length = []
        for edge in pdf_doc.edges[:100]:
            if edge['orientation'] == "v":
                continue

            if edge["width"] >= max_textline_length:
                edges_greater_than_textline_length.append(edge)

        for rect in pdf_doc.rects[:100]:
            if rect['height'] > 2:
                continue
            
            if rect['width'] >= max_textline_length:
                rects_greater_than_textline_length.append(rect)

        for element in sorted(text, key=lambda x: x["top"])[:20]:
            #print(element)
            if str.isnumeric(element['text']) or (): # search for pagenum
                crop_from_top = int(element['bottom']) + 1
                break

        top_crops.append(crop_from_top)
        crop_from_top = None

        for element in sorted(text, key=lambda x: x["top"], reverse=True)[:20]:
            #print(element)
            if str.isnumeric(element['text']): # search for pagenum
                crop_from_bottom = int(page_height - element['top']) - 1
                break    
        bottom_crops.append(crop_from_bottom)
        crop_from_bottom = None

    boolean_crop_top, boolean_crop_bottom = [], []
    for i,t in enumerate(top_crops):
        if t is not None:
            boolean_crop_top.append(t == top_crops[0])
        else:
            if i % 3 == 0:
                boolean_crop_top.append(True)
            else:
                boolean_crop_top.append(False)

    for i,t in enumerate(bottom_crops):
        if t is not None:
            boolean_crop_bottom.append(t == bottom_crops[0])
        else:
            if i % 3 == 0:
                boolean_crop_bottom.append(True)
            else:
                boolean_crop_bottom.append(False)

    #boolean_crop_top = [t == top_crops[0] if t is not None else 5/(i+1) for i, t in enumerate(top_crops)]
    #boolean_crop_bottom = [t == bottom_crops[0] if t is not None else 5/(i+1) for i, t in enumerate(bottom_crops)]
    # if all(boolean_crop_top) and any(boolean_crop_bottom):
    #     crop_from_bottom = 0
    #     crop_from_top = top_crops[0]
    # elif any(boolean_crop_top) and all(boolean_crop_bottom):
    #     crop_from_top = 0
    #     crop_from_bottom = bottom_crops[0]
    # elif any(boolean_crop_top) and any(boolean_crop_bottom):
    #     crop_from_top = int(FIXED_CROP_TOP*page_height)
    #     crop_from_bottom = page_height - int(FIXED_CROP_BOTTOM*page_height)
    if boolean_crop_top.count(True) >= len(boolean_crop_top) / 2 and boolean_crop_bottom.count(True) < len(boolean_crop_bottom) / 2:
        crop_from_bottom = 0
        crop_from_top = top_crops[0]
    elif boolean_crop_top.count(True) < len(boolean_crop_top) / 2 and boolean_crop_bottom.count(True) >= len(boolean_crop_bottom) / 2:
        crop_from_top = 0
        crop_from_bottom = bottom_crops[0]
    elif boolean_crop_top.count(True) < len(boolean_crop_top) / 2 and boolean_crop_bottom.count(True) < len(boolean_crop_bottom) / 2:
        crop_from_top = int(FIXED_CROP_TOP*page_height)
        crop_from_bottom = page_height - int(FIXED_CROP_BOTTOM*page_height)
    else:
        crop_from_top = top_crops[0] if top_crops[0] is not None else top_crops[1]
        crop_from_bottom = bottom_crops[0] if bottom_crops[0] is not None else bottom_crops[1]


    if len(edges_greater_than_textline_length) > 0:
        max_y1 = 0
        for edge in edges_greater_than_textline_length:
            if edge["y1"] > max_y1:
                max_y1 = edge["y1"]
        
        if int(page_height - max_y1) > crop_from_top:
            crop_from_top = int(page_height - max_y1)
    
    if len(rects_greater_than_textline_length) > 0:
        max_y1 = 0
        for rect in rects_greater_than_textline_length:
            if rect["y1"] > max_y1:
                max_y1 = rect["y1"]
        
        if int(page_height - max_y1) > crop_from_top:
            crop_from_top = int(page_height - max_y1)

    return crop_from_top, crop_from_bottom


for pdf_file in os.listdir(pdf_path):
    #if pdf_file != "diss_Senn.pdf":
    #    continue
    # if pdf_file != "_Dissertation_Johannes_Schildgen.pdf":
    #    continue

    print(f"processing pdf file {pdf_file}")
    filtered_text = ""  
    filepath = os.path.join(pdf_path, pdf_file)

    # parser = PDFParser(open(filepath, 'rb'))
    # Create a PDF document object that stores the document structure.
    # Supply the password for initialization.
    # document = PDFDocument(parser, '')
    # outlines = document.get_outlines()
    # for (level,title,dest,a,se) in outlines:
    #    print(level, title)
    pdf_doc = pdfplumber.open(filepath)
    crop_time_start = time.time()
    crop_from_top, crop_from_bottom = get_crop_height(pdf_doc)
    crop_time_end = time.time()
    for page in pdf_doc.pages:
        cropped_page = page.crop([0, crop_from_top, page.width, (page.height-crop_from_bottom)], relative=False)
        img = cropped_page.to_image()
        img.show()
        text = cropped_page.extract_text(x_tolerance=1.5)
        if pattern.search(text):
            continue
        filtered_text += text
        pages += 1
    # fulltext = extract_text(filepath)   
    with open(os.path.join(output_path, pdf_file.replace(".pdf", ".txt")), "w") as file:
        file.write(filtered_text)    


end_time = time.time()
with open(os.path.join(output_path, "processing_information.txt"), "w") as file:
    file.write(f"The processing time took: {end_time-start_time} seconds to process {len(os.listdir(pdf_path))} PDFs with {pages} pages.\n The calculation of the page cropping took {crop_time_end-crop_time_start} seconds.")