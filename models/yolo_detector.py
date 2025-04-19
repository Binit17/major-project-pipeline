import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

class YOLODetector:
    def __init__(self, model_path='models/yolo.pt'):
        # Initialize the YOLO model
        self.model = YOLO(model_path)
        
        # Check for GPU availability and move model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.to('cuda')
            print(f"YOLOv8 using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("YOLOv8 using CPU")

    def detect_text(self, image):
        """Detect text regions in an image with GPU acceleration"""
        # Free up GPU memory before detection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Explicitly specify device for prediction
        results = self.model.predict(image, device=self.device)
        boxes = [box.xyxy.cpu().numpy() for result in results for box in result.boxes]
        return boxes

    def draw_bounding_boxes(self, image_path, output_path="output.jpg"):
        """Detect text and draw bounding boxes"""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        boxes = self.detect_text(image)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[0])
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        image.save(output_path)
        image.show()  # Show the image with bounding boxes
