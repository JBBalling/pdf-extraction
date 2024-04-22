import os
import pdfplumber
import re
import layoutparser as lp

from pdf2image import convert_from_path



pdf_path = "pdf-data"
pattern = re.compile(r"(.*verzeichnis(\n|$)|.*übersicht(\n|$)|.*\.\.\.\.|.*\. \. \. \.)")
model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])

for pdf_file in os.listdir(pdf_path):
    #if pdf_file != "diss_Senn.pdf":
    #    continue
    # if pdf_file != "_Dissertation_Johannes_Schildgen.pdf":
    #    continue

    print(f"processing pdf file {pdf_file}")
    filtered_text = ""  
    filepath = os.path.join(pdf_path, pdf_file)
    images = convert_from_path(filepath, 200, first_page=0, last_page=30)


    category = [3, 7, 9, 10]
    pdf_doc = pdfplumber.open(filepath)
    for image in images:
        layout = model.detect(image)
        layouted_img = lp.visualization.draw_box(image, layout)
        layouted_img.show()
        print("s")