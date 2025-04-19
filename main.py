import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from models.yolo_detector import YOLODetector
from models.resnet_classifier import ResNetClassifier
from models.trocr_ocr import TrOCROCR
from utils.image_processing import preprocess_image
from utils.box_sorting import sort_boxes

#define the accelerator here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class OCRPipeline:
    def __init__(self):
        self.yolo_detector = YOLODetector()
        self.resnet_classifier = ResNetClassifier(device)
        self.trocr_ocr = TrOCROCR(device)

    def process_image(self, image_path, visualize=True):
        """Main OCR processing pipeline with batch processing"""
        # Clear GPU memory at the start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        orig_image = Image.open(image_path).convert("RGB")
        img_with_boxes = orig_image.copy()

        # Detect text regions with YOLO
        all_boxes = self.yolo_detector.detect_text(orig_image)
        sorted_boxes = sort_boxes([box for sublist in all_boxes for box in sublist])

        # Prepare for batch processing
        cropped_regions = []
        classifier_inputs = []
        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_region = orig_image.crop((x1, y1, x2, y2))
            cropped_regions.append(cropped_region)
            classifier_inputs.append(preprocess_image(cropped_region))
        
        # Batch classify all regions at once
        classification_results = self.resnet_classifier.classify_batch(classifier_inputs)
        
        # Collect valid (non-strike-through) regions for OCR
        valid_regions = []
        valid_indices = []
        box_classifications = []
        
        for i, ((class_id, confidence), box) in enumerate(zip(classification_results, sorted_boxes)):
            x1, y1, x2, y2 = map(int, box)
            box_classifications.append((x1, y1, x2, y2, class_id, confidence))
            
            if class_id == 0:  # Not strike-through
                valid_regions.append(cropped_regions[i])
                valid_indices.append(i)
        
        # Initialize final text as empty strings for all boxes
        final_text = [''] * len(sorted_boxes)
        
        # Batch process OCR for valid regions
        if valid_regions:
            try:
                ocr_results = self.trocr_ocr.recognize_batch(valid_regions)
                
                # Map OCR results back to their original positions
                for idx, text in zip(valid_indices, ocr_results):
                    final_text[idx] = text
            except RuntimeError as e:
                # If batch OCR fails, try individual processing
                print(f"Batch OCR processing failed: {str(e)}")
                for i, region in zip(valid_indices, valid_regions):
                    try:
                        text = self.trocr_ocr.recognize_text(region)
                        final_text[i] = text
                    except Exception as ex:
                        print(f"Error processing region {i}: {str(ex)}")
        
        # Filter out empty strings
        clean_text = [t for t in final_text if t]
        
        # Visualization
        if visualize:
            draw = ImageDraw.Draw(img_with_boxes)
            for x1, y1, x2, y2, class_id, conf in box_classifications:
                color = "green" if class_id == 0 else "red"
                label = f"{'Text' if class_id == 0 else 'Strike'} ({conf:.2f})"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1+5, y1-25), label, fill=color)

            plt.figure(figsize=(16, 12))
            plt.imshow(img_with_boxes)
            plt.title("OCR Results - Green: Valid Text, Red: Strike-through", fontsize=18, pad=20)
            plt.axis('off')
            plt.show()

        # Clear GPU memory at the end
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return ' '.join(clean_text)


if __name__ == "__main__":
    pipeline = OCRPipeline()
    extracted_text = pipeline.process_image("test_images/try1.jpg")
    print("Extracted Text:", extracted_text)
