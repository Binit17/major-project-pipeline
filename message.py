import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

from models.yolo_detector import YOLODetector
from models.resnet_classifier import ResNetClassifier
# from models.trocr_ocr import TrOCROCR
from utils.image_processing import preprocess_image, updated_preprocess_image
from utils.box_sorting import sort_boxes

# Define the accelerator here
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('Dataset', exist_ok=True)

# Create unique folder names
i = 1
while os.path.exists(f'Dataset/nonstrike{i}'):
    i += 1
nonstrike_folder = f'nonstrike{i}'

i = 1
while os.path.exists(f'Dataset/strike{i}'):
    i += 1
strike_folder = f'strike{i}'

# Create directories
os.makedirs(f'Dataset/{nonstrike_folder}', exist_ok=True)
os.makedirs(f'Dataset/{strike_folder}', exist_ok=True)

class OCRPipeline:
    def __init__(self):
        self.yolo_detector = YOLODetector()
        self.resnet_classifier = ResNetClassifier(device)
        # self.trocr_ocr = TrOCROCR(device)

    def process_image(self, image_path, visualize=True):
        """Main OCR processing pipeline"""
        orig_image = Image.open(image_path).convert("RGB")
        
        # Get bounding boxes from YOLO detector
        all_boxes = self.yolo_detector.detect_text(orig_image)
        
        # Flatten the list of boxes
        sorted_boxes = [box for sublist in all_boxes for box in sublist]

        final_text = []
        classification_results = []
        
        # Process each detected box
        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_region = orig_image.crop((x1, y1, x2, y2))

            # Preprocess and classify the cropped image
            classifier_input = updated_preprocess_image(cropped_region)
            class_id, confidence = self.resnet_classifier.classify(classifier_input)
            classification_results.append((x1, y1, x2, y2, class_id, confidence))
            
            # Save the cropped image to the appropriate folder
            if class_id == 0:
                classifier_input.save(f'Dataset/{nonstrike_folder}/cropped_{x1}_{y1}.png')
            else:
                classifier_input.save(f'Dataset/{strike_folder}/cropped_{x1}_{y1}.png')
            
            # If it's regular text (not strikethrough), could run OCR here
            # if class_id == 0:
            #     ocr_text = self.trocr_ocr.recognize_text(cropped_region)
            #     final_text.append(ocr_text)

        # Visualization - only one image with boxes
        if visualize:
            img_with_boxes = orig_image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            for x1, y1, x2, y2, class_id, conf in classification_results:
                color = "green" if class_id == 0 else "red"
                label = f"{'Text' if class_id == 0 else 'Strike'} ({conf:.2f})"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1+5, y1-25), label, fill=color)

            plt.figure(figsize=(16, 12))
            plt.imshow(img_with_boxes)
            plt.title("OCR Results - Green: Valid Text, Red: Strike-through", fontsize=18, pad=20)
            plt.axis('off')
            plt.show()

        return ' '.join(final_text)

if __name__ == "__main__":
    pipeline = OCRPipeline()
    extracted_text = pipeline.process_image("test_images/9.jpg")
    print("Extracted Text:", extracted_text)