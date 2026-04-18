import json
import os
import cv2  # OpenCV library to read image sizes

# --- CONFIGURATION ---
INPUT_FILE = 'project_export.json'   # The file you downloaded from Label Studio
IMAGE_FOLDER = 'Ground Truth(Test case)'           # The folder containing your .jpg invoices
OUTPUT_FILE = 'ground_truth_master.json' # The final "Answer Key" file

def convert_label_studio_to_master_gt():
    # 1. Load the Label Studio data
    with open(INPUT_FILE, 'r') as f:
        ls_data = json.load(f)

    master_ground_truth = []

    print(f"Processing {len(ls_data)} documents...")

    # 2. Loop through every labeled document
    for task in ls_data:
        # Label Studio stores the filename in 'data' -> 'image'
        # It often looks like "/data/upload/1/invoice_001.jpg", so we get just the name.
        raw_filename = task['data']['image']
        filename = os.path.basename(raw_filename)

        # 3. Get the Image Dimensions (Crucial for % -> Pixel conversion)
        img_path = os.path.join(IMAGE_FOLDER, filename)
        
        if not os.path.exists(img_path):
            print(f"⚠ Warning: Image '{filename}' not found in folder. Skipping.")
            continue
            
        # Read image to get (Height, Width)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Error: Could not read '{filename}'. Corrupt file?")
            continue
            
        img_height, img_width = img.shape[:2]

        # 4. Initialize the Clean Entry for this Document
        doc_entry = {
            "doc_id": filename,
            "fields": {
                # Initialize empty defaults just in case you missed a label
                "dealer_name": None,
                "model_name": None,
                "horse_power": None,
                "asset_cost": None,
                "signature": {"present": False, "bbox": []},
                "stamp": {"present": False, "bbox": []}
            }
        }

        # 5. Extract the Labels (The Annotations)
        # 'annotations' is a list (in case multiple people labeled it). We take the most recent one [-1].
        if not task.get('annotations'):
            continue
            
        labels = task['annotations'][-1]['result']

        for label in labels:
            # Get the Label Name (e.g., "Dealer Name", "Dealer Signature")
            # Note: Label Studio creates a list, so we take the first item [0]
            tag_name = label['value']['rectanglelabels'][0]

            # --- MATH: Convert Percentages to Pixels ---
            # Label Studio gives: x, y, width, height (0-100 scale)
            x_pct = label['value']['x']
            y_pct = label['value']['y']
            w_pct = label['value']['width']
            h_pct = label['value']['height']

            # Math: (Percentage / 100) * Total_Pixels
            x1 = int((x_pct / 100) * img_width)
            y1 = int((y_pct / 100) * img_height)
            w_px = int((w_pct / 100) * img_width)
            h_px = int((h_pct / 100) * img_height)
            
            # Convert to [x_min, y_min, x_max, y_max] format
            x2 = x1 + w_px
            y2 = y1 + h_px
            final_bbox = [x1, y1, x2, y2]

            # --- LOGIC: Store Data based on Field Type ---
            
            # Map the Label Studio name to the JSON key required by Hackathon
            if tag_name == "Dealer Signature":
                doc_entry["fields"]["signature"] = {
                    "present": True,
                    "bbox": final_bbox
                }
            elif tag_name == "Dealer Stamp":
                doc_entry["fields"]["stamp"] = {
                    "present": True,
                    "bbox": final_bbox
                }
            else:
                # Text Fields (Name, Model, HP, Cost)
                # For these, we need the text you TYPED in Label Studio
                user_typed_text = label['value'].get('text', [""])[0]
                
                # Standardize keys
                key_map = {
                    "Dealer Name": "dealer_name",
                    "Model Name": "model_name",
                    "Horse Power": "horse_power",
                    "Asset Cost": "asset_cost"
                }
                
                if tag_name in key_map:
                    json_key = key_map[tag_name]
                    
                    # Special Clean-up for Numbers
                    if json_key == "horse_power" or json_key == "asset_cost":
                        # Ensure it's a number (remove "HP", "Rs", commas)
                        clean_text = ''.join(filter(str.isdigit, str(user_typed_text)))
                        # Convert to integer if not empty
                        final_val = int(clean_text) if clean_text else 0
                    else:
                        final_val = user_typed_text

                    doc_entry["fields"][json_key] = final_val

        master_ground_truth.append(doc_entry)

    # 6. Save the Final File
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(master_ground_truth, f, indent=4)
    
    print(f"✅ Success! Created '{OUTPUT_FILE}' with {len(master_ground_truth)} documents.")

if __name__ == "__main__":
    convert_label_studio_to_master_gt()