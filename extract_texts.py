import pdfplumber
import os
from unidecode import unidecode

def extract_texts_from_pdfs(base_folder="data"):
    extracted_texts = []

    # Walk through all subfolders and files
    for root, subdirs, files in os.walk(base_folder):
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)
                category = os.path.basename(root)  # ex: codes, laws, official_gazettes
                print(f"Reading: {filename} ({category})")
 
                try:
                    with pdfplumber.open(file_path) as pdf:
                        full_text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                full_text += unidecode(page_text) + "\n"

                        extracted_texts.append({
                            "filename": filename,
                            "category": category,
                            "text": full_text
                        })
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return extracted_texts
