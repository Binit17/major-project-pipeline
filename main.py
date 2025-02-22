import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from models.yolo_detector import YOLODetector
from models.resnet_classifier import ResNetClassifier
from models.trocr_ocr import TrOCROCR
from utils.image_processing import preprocess_image
from utils.box_sorting import sort_boxes

#define the accelerator here
device = torch.device("mps" if torch.cuda.is_available() else "cpu") 

class OCRPipeline:
    def __init__(self):
        self.yolo_detector = YOLODetector()
        self.resnet_classifier = ResNetClassifier(device)
        self.trocr_ocr = TrOCROCR(device)

    def process_image(self, image_path, visualize=True):
        """Main OCR processing pipeline"""
        orig_image = Image.open(image_path).convert("RGB")
        img_with_boxes = orig_image.copy()

        #! Uncomment below if you want to visualize the input image.
        # if visualize:
        #     plt.figure(figsize=(12, 8))
        #     plt.imshow(orig_image)
        #     plt.title("Input Image", fontsize=16)
        #     plt.axis('off')
        #     plt.show()

        # Detect text regions with YOLO
        all_boxes = self.yolo_detector.detect_text(orig_image)
        # sorted_boxes = sort_boxes(all_boxes)
        #! Here the boxes needs to be sorted later.
        # sorted_boxes = all_boxes
        sorted_boxes = [box for sublist in all_boxes for box in sublist]


        final_text = []
        classification_results = []
        draw = ImageDraw.Draw(img_with_boxes)

        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_region = orig_image.crop((x1, y1, x2, y2))
            classifier_input = preprocess_image(cropped_region)

            class_id, confidence = self.resnet_classifier.classify(classifier_input)
            classification_results.append((x1, y1, x2, y2, class_id, confidence))

            if class_id == 0:
                ocr_text = self.trocr_ocr.recognize_text(cropped_region)
                final_text.append(ocr_text)

        # Visualization
        if visualize:
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
    extracted_text = pipeline.process_image("test_images/try2.jpg")
    print("Extracted Text:", extracted_text)
