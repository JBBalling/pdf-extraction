import os
import time
from pdfminer.high_level import extract_text


start_time = time.time()
pdf_path = "pdf-data"
output_path = "pdfminer-output"


for pdf_file in os.listdir(pdf_path):
    filepath = os.path.join(pdf_path, pdf_file)
    fulltext = extract_text(filepath)   
    with open(os.path.join(output_path, pdf_file.replace(".pdf", ".txt")), "w") as file:
        file.write(fulltext)    


end_time = time.time()
with open(os.path.join(output_path, "processing_information.txt"), "w") as file:
    file.write(f"The processing time took: {end_time-start_time} seconds to process {len(os.listdir(pdf_path))} PDFs.")