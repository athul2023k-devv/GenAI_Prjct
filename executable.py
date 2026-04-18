import sys
import os
import json
import cv2
import re
import numpy as np
from paddleocr import PaddleOCR
from collections import Counter

def detect_signature_and_stamp_math(img, img_name):
    height, width = img.shape[:2]
    
    sig_zone_y = int(height * 0.70)
    sig_zone_x = int(width * 0.40)
    
    roi = img[sig_zone_y:height, sig_zone_x:width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    ink_pixels = cv2.countNonZero(binary)
    sig_result = {"present": False, "bbox": []}
    
    if ink_pixels > 400:
        coords = cv2.findNonZero(binary)
        x, y, w, h = cv2.boundingRect(coords)
        real_x, real_y = sig_zone_x + x, sig_zone_y + y
        sig_result = {
            "present": True, 
            "bbox": [real_x, real_y, real_x + w, real_y + h]
        }

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    _, sat_mask = cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
    
    stamp_pixels = cv2.countNonZero(sat_mask)
    stamp_result = {"present": False, "bbox": []}
    
    if stamp_pixels > 250:
        coords = cv2.findNonZero(sat_mask)
        x, y, w, h = cv2.boundingRect(coords)
        real_x, real_y = sig_zone_x + x, sig_zone_y + y
        stamp_result = {
            "present": True, 
            "bbox": [real_x, real_y, real_x + w, real_y + h]
        }

    return sig_result, stamp_result

def extract_field_logic(all_text_lines):
    data = {"dealer_name": None, "model_name": None, "horse_power": None, "asset_cost": None}
    all_found_costs = []
    
    for i, line in enumerate(all_text_lines):
        text = line[1][0]
        text_lower = text.lower()
        
        if data["dealer_name"] is None:
            if any(x in text_lower for x in ['motors', 'auto', 'sales', 'agency', 'pvt', 'ltd']):
                if "gstin" not in text_lower and "@" not in text and ".com" not in text_lower:
                    data["dealer_name"] = text

        if data["model_name"] is None:
            brands = ["MARUTI", "HYUNDAI", "TATA", "MAHINDRA", "TOYOTA", "KIA", "HONDA", "MG", "SKODA", "NISSAN"]
            for b in brands:
                if b in text.upper():
                    data["model_name"] = text
                    break
            if data["model_name"] is None and ("model" in text_lower or "variant" in text_lower):
                 data["model_name"] = text

        if data["horse_power"] is None:
            if "hp" in text_lower or "bhp" in text_lower:
                match = re.search(r'\b(\d{2})\s*(?:hp|bhp|ps)\b', text_lower)
                if match:
                    data["horse_power"] = int(match.group(1))

        clean_line = re.sub(r'[^\d]', '', text)
        if 6 <= len(clean_line) <= 7:
            val = int(clean_line)
            if val > 50000:
                all_found_costs.append(val)

    if all_found_costs:
        counts = Counter(all_found_costs)
        most_common = counts.most_common()
        candidate, frequency = most_common[0]
        
        if frequency >= 2:
            data["asset_cost"] = candidate
        else:
            data["asset_cost"] = max(all_found_costs)

    return data

def process_documents():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'sample_output')
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'result.json')
    
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

    input_files = []
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.isfile(input_path):
            input_files.append(input_path)
        elif os.path.isdir(input_path):
            for f in os.listdir(input_path):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    input_files.append(os.path.join(input_path, f))
    else:
        default_folder = os.path.join(BASE_DIR, 'Ground Truth (Test case)')
        if os.path.exists(default_folder):
            for f in os.listdir(default_folder):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    input_files.append(os.path.join(default_folder, f))

    print(f"Processing {len(input_files)} inputs...")
    results = []

    for img_path in input_files:
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_name = os.path.basename(img_path)
            print(f"Scanning: {img_name}")
            
            doc_result = {"doc_id": img_name, "fields": {}}

            sig_data, stamp_data = detect_signature_and_stamp_math(img, img_name)
            doc_result["fields"]["signature"] = sig_data
            doc_result["fields"]["stamp"] = stamp_data

            ocr_output = ocr_engine.ocr(img_path)
            text_data = {"dealer_name": None, "model_name": None, "horse_power": None, "asset_cost": None}
            
            if ocr_output and ocr_output[0]:
                text_data = extract_field_logic(ocr_output[0])
                
            for k, v in text_data.items():
                doc_result["fields"][k] = {"value": v, "bbox": []}
            
            results.append(doc_result)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_documents()