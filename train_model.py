from ultralytics import YOLO
import os

def train_yolo():
    # 1. Load the Model
    # 'yolov8n.pt' is the "Nano" version. It is small, fast, and perfect for this task.
    # It will automatically download if you don't have it.
    model = YOLO('yolov8n.pt') 
    
    print("🚀 Starting Training... This might take 5-10 minutes.")
    
    # 2. Start Training
    # data='data.yaml' -> Points to the map we made in Step 3
    # epochs=50 -> How many times it loops over the data
    # imgsz=640 -> Resizes images to standard square for speed
    results = model.train(data='data.yaml', epochs=50, imgsz=640)
    
    print("-" * 30)
    print("🎉 TRAINING FINISHED!")
    print(f"💾 Your new AI model is saved at: {results.save_dir}/weights/best.pt")
    print("Use 'best.pt' for your final hackathon submission.")

if __name__ == '__main__':
    train_yolo()