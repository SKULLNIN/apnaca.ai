import requests
import fitz  # PyMuPDF
import re
import json

# Step 1: Download the PDF File
pdf_url = "https://www.incometax.gov.in/iec/foportal/sites/default/files/2024-04/CBDT_e-Filing_ITR%201_Validation%20Rules_AY2024-25_V1.0..pdf"
response = requests.get(pdf_url)
pdf_file_path = "downloaded_pdf.pdf"

if response.status_code == 200:
    with open(pdf_file_path, 'wb') as pdf_file:
        pdf_file.write(response.content)
    print("PDF file downloaded successfully.")
else:
    print(f"Failed to download the PDF file. Status code: {response.status_code}")
    exit()

# Extract Text from the PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text("text") for page in doc)  # Extract text with layout preservation
    return text

extracted_text = extract_text_from_pdf(pdf_file_path)
print("Text extracted from PDF.")

# Main Program for  Clean and Preprocess the Extracted Text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Add space between letters and numbers  ( main shiit)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Add space between numbers and letters  ( main shiit)
    return text.strip()

cleaned_text = clean_text(extracted_text)
print("Text cleaned successfully.")

# Save the Cleaned Data ( baad me change karuga)
with open("cleaned_text.txt", 'w', encoding='utf-8') as txt_file:
    txt_file.write(cleaned_text)
print("Cleaned text saved to cleaned_text.txt.")

# Prepare Data for Fine-Tuning (JSONL Format)  ( sample )
fine_tune_data = [{"question": "What are the validation rules for ITR 1?", "answer": cleaned_text}]

with open("fine_tune_data.jsonl", 'w', encoding='utf-8') as jsonl_file:
    for entry in fine_tune_data:
        jsonl_file.write(json.dumps(entry) + "\n")

print("Data prepared for fine-tuning and saved to fine_tune_data.jsonl.")
