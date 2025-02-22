import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """Preprocess image for ResNet classification"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binarized = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )
    return Image.fromarray(binarized).convert("RGB")
