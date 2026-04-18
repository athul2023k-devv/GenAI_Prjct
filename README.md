# Intelligent Document AI for Invoice Field Extraction

An end-to-end, low-latency, and cost-efficient Document AI system designed to extract structured fields from highly variable tractor loan quotations.

# 1. System Architecture & Pipeline

Our solution employs a Hybrid Architecture that decouples visual layout understanding from textual semantic extraction to optimize both cost and speed:

   **Ingestion:** Converts PDFs/images to standardized formats.
    **Visual Detection (YOLOv8-Nano):** A lightweight object detection model fine-tuned to identify the presence and bounding boxes of Dealer Signatures and Stamps. Chosen for computational efficiency compared to heavy Vision-Language Models.
    **Text Extraction (PaddleOCR):** Processes the document to extract text across all the invoices.
    **Entity Structuring & Reasoning:**
        **Deterministic Logic (Regex):** Extracts structurally rigid numeric fields like Asset Cost and Horse Power.
        **Fuzzy Search (rapidfuzz):** Maps extracted OCR strings against a master database to identify Dealer Name and Model Name with a >=90% confidence threshold.

# 2. Handling the Ground Truth Strategy
To establish a rigorous evaluation benchmark and train the visual models, we utilized a manual annotation strategy. We curated and annotated a representative subset of the data to create a high-quality ground_truth.json file. This served as our foundation for training the YOLO model and performing self-consistency evaluations across our validation split.

# 3. Execution Instructions

**Prerequisites:**
pip install -r requirements.txt