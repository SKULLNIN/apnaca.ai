import os
import re
import shutil
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from typing import Dict, List, Optional
import logging
import tempfile

# Configuration
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE_MB = 5
UPLOAD_FOLDER = tempfile.mkdtemp()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better OCR results"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        return cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def extract_text_from_image(image_path: str) -> Optional[str]:
    """Robust text extraction with error handling"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image file")
            
        processed_image = preprocess_image(image)
        custom_config = r'--oem 3 --psm 6 -l eng+ind'
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return None

def validate_gstin(gstin: str) -> bool:
    """Enhanced GSTIN validation using checksum"""
    if len(gstin) != 15:
        return False
    
    checksum = 0
    weights = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    
    try:
        for i in range(14):
            char = gstin[i]
            value = int(char) if char.isdigit() else (ord(char.upper()) - 55)
            checksum += value * weights[i]
        
        check_digit = str((checksum % 11) % 10)
        return check_digit == gstin[-1]
    except:
        return False

def extract_financial_data(text: str) -> Dict:
    """Advanced pattern matching with validation"""
    patterns = {
        'gstin': (r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d[Z]{1}[A-Z\d]{1}\b', validate_gstin),
        'invoice_date': (
            r'\b(0[1-9]|[12][0-9]|3[01])[-/](0[1-9]|1[012])[-/](20[2-9][0-9])\b',
            lambda d: True
        ),
        'total_amount': (
            r'(?i)(?:total|grand total|amount payable)\s*[:]?\s*([₹$]?[\d,]+\.\d{2})',
            lambda a: float(a.replace(',', '').replace('₹', '').replace('$', '')) > 0
        )
    }

    results = {}
    for key, (pattern, validator) in patterns.items():
        matches = re.findall(pattern, text)
        valid_matches = [m for m in matches if validator(m[0] if isinstance(m, tuple) else m)]
        results[key] = valid_matches[0] if valid_matches else None

    return results
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "valid_endpoints": ["/api/v1/extract"]
    }), 404
    
@app.route('/api/v1/extract', methods=['POST'])
def handle_invoice_extraction():
    """Secure endpoint for invoice processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400


    try:
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)

        text_content = []
        if filename.lower().endswith('.pdf'):
            images = convert_from_path(file_path, poppler_path=r'C:\path\to\poppler-xx\bin')
            for page_num, image in enumerate(images):
                page_path = os.path.join(temp_dir, f'page_{page_num}.jpg')
                image.save(page_path, 'JPEG')
                page_text = extract_text_from_image(page_path)
                if page_text:
                    text_content.append(page_text)
        else:
            page_text = extract_text_from_image(file_path)
            if page_text:
                text_content.append(page_text)

        if not text_content:
            return jsonify({'error': 'No extractable content found'}), 400

        full_text = '\n'.join(text_content)
        extracted_data = extract_financial_data(full_text)
        
        return jsonify({
            'status': 'success',
            'data': extracted_data,
            'metadata': {
                'pages_processed': len(text_content),
                'text_length': len(full_text)
            }
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Processing failed', 'detail': str(e)}), 500
        
    finally:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')
