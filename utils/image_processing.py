# import cv2
# import numpy as np
# from PIL import Image

# def preprocess_image(image):
#     """Preprocess image for ResNet classification"""
#     gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     binarized = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 21, 10
#     )
#     return Image.fromarray(binarized).convert("RGB")


import cv2
import numpy as np
from PIL import Image, ImageOps

# def preprocess_image(image, target_size=(224, 224)):
#     """
#     Preprocess text image with resize first and padding after preprocessing.

#     Parameters:
#         image (PIL.Image or numpy.ndarray): Input image in BGR format or PIL format.
#         target_size (tuple): The target size for resizing while maintaining aspect ratio.

#     Returns:
#         PIL.Image: The preprocessed (binarized) image with black text on white background.
#     """
#     # Convert PIL Image to numpy array if necessary
#     if isinstance(image, Image.Image):
#         image = np.array(image)

#     # Convert BGR to RGB and then to grayscale
#     if len(image.shape) == 3:
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
#     else:
#         image_gray = image  # Already grayscale

#     # Convert to PIL image and resize while maintaining aspect ratio
#     image_pil = Image.fromarray(image_gray)
#     image_pil.thumbnail(target_size, Image.Resampling.LANCZOS)

#     # Convert back to numpy array
#     image_np = np.array(image_pil, dtype=np.float32)

#     # Apply Gaussian blur to reduce noise
#     # image_np = cv2.GaussianBlur(image_np, (3, 3), 0)

#     # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
#     image_np = clahe.apply(image_np.astype(np.uint8))

#     # Calculate mean intensity for dynamic thresholding
#     mean_intensity = np.mean(image_np)
    
#     # Define adaptive thresholding parameters based on mean intensity
#     block_size = 15 if mean_intensity > 127 else 11
#     C = 10 if mean_intensity > 127 else 2

#     # Apply adaptive thresholding
#     image_binarized = cv2.adaptiveThreshold(
#         image_np,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         block_size,
#         C
#     )

#     # Apply morphological operations to clean up noise
#     kernel = np.ones((2, 2), np.uint8)
#     image_binarized = cv2.morphologyEx(image_binarized, cv2.MORPH_CLOSE, kernel)

#     # Convert back to PIL image for padding
#     image_pil = Image.fromarray(image_binarized)

#     # Apply padding to maintain size
#     image_padded = ImageOps.pad(image_pil, target_size, method=Image.Resampling.NEAREST, color=255)

#     # Convert to RGB with black text on white background
#     image_rgb = ImageOps.colorize(image_padded, black="black", white="white")

#     return image_rgb



# import cv2
# import numpy as np
# from PIL import Image, ImageOps

# def preprocess_image(image, target_size=(224, 224)):
#     """
#     Preprocess text image with resize first and padding after preprocessing.

#     Parameters:
#         image (PIL.Image or numpy.ndarray): Input image in BGR format or PIL format.
#         target_size (tuple): The target size for resizing while maintaining aspect ratio.

#     Returns:
#         PIL.Image: The preprocessed (binarized) image with black text on white background.
#     """
#     # Convert PIL Image to numpy array if necessary
#     if isinstance(image, Image.Image):
#         image = np.array(image)

#     # Convert BGR to RGB and then to grayscale
#     if len(image.shape) == 3:
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
#     else:
#         image_gray = image  # Already grayscale

#     # Convert to PIL image and resize while maintaining aspect ratio
#     image_pil = Image.fromarray(image_gray)
#     image_pil.thumbnail(target_size, Image.Resampling.LANCZOS)

#     # Convert back to numpy array
#     image_np = np.array(image_pil, dtype=np.uint8)

#     # Calculate mean intensity for dynamic thresholding
#     mean_intensity = np.mean(image_np)

#     # Define adaptive thresholding parameters based on mean intensity
#     block_size = 15 if mean_intensity > 127 else 11
#     C = 10 if mean_intensity > 127 else 2

#     # Apply adaptive thresholding
#     image_binarized = cv2.adaptiveThreshold(
#         image_np,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         block_size,
#         C
#     )

#     # Apply morphological operations to clean up noise
#     kernel = np.ones((2, 2), np.uint8)
#     image_binarized = cv2.morphologyEx(image_binarized, cv2.MORPH_CLOSE, kernel)

#     # Convert back to PIL image for padding
#     image_pil = Image.fromarray(image_binarized)

#     # Apply padding to maintain size
#     image_padded = ImageOps.pad(image_pil, target_size, method=Image.Resampling.NEAREST, color=255)

#     # Convert to RGB with black text on white background
#     image_rgb = ImageOps.colorize(image_padded, black="black", white="white")

#     return image_rgb


import cv2
import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess text image with improved adaptive thresholding for zoomed small text.

    Parameters:
        image (PIL.Image or numpy.ndarray): Input image.
        target_size (tuple): Target size for resizing.

    Returns:
        PIL.Image: The processed binarized image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert BGR to grayscale
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Resize while maintaining aspect ratio
    image_pil = Image.fromarray(image_gray)
    image_pil.thumbnail(target_size, Image.BILINEAR)

    # Convert to numpy array
    image_np = np.array(image_pil, dtype=np.uint8)

    # Estimate text density (ratio of text pixels to total pixels)
    text_pixels = np.sum(image_np < 200)  # Count dark pixels
    total_pixels = image_np.shape[0] * image_np.shape[1]
    text_density = text_pixels / total_pixels  # Compute density

    # Dynamically adjust block size based on text density
    if text_density < 0.02:  # Very small zoomed-in text
        block_size = 7
        C = 5
    elif text_density < 0.1:  # Medium text
        block_size = 11
        C = 7
    else:  # Large text
        block_size = 15
        C = 10

    # Choose thresholding method based on text density
    if text_density < 0.015:  # Very low-density text (switch to Otsu)
        _, image_binarized = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        image_binarized = cv2.adaptiveThreshold(
            image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )

    # Apply Morphological Opening to remove small noise dots
    kernel = np.ones((2, 2), np.uint8)
    image_binarized = cv2.morphologyEx(image_binarized, cv2.MORPH_CLOSE, kernel)

    # Convert to PIL Image and pad to target size
    image_pil = Image.fromarray(image_binarized)
    image_padded = ImageOps.pad(image_pil, target_size, method=Image.Resampling.NEAREST, color=255)

    # Convert to RGB with black text on white background
    image_rgb = ImageOps.colorize(image_padded, black="black", white="white")

    return image_rgb



def updated_preprocess_image(image, target_size=(224, 224), crop_percentage=0.05):
    """
    Preprocess text image with manual border cropping and Otsu thresholding.

    Parameters:
        image (PIL.Image or numpy.ndarray): Input image.
        target_size (tuple): Target size for resizing.
        crop_percentage (float): Percentage to crop from all sides (0.05 = 5%)

    Returns:
        PIL.Image: The processed binarized image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert BGR to grayscale
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Resize while maintaining aspect ratio
    image_pil = Image.fromarray(image_gray)
    image_pil.thumbnail(target_size, Image.BILINEAR)
    image_np = np.array(image_pil, dtype=np.uint8)
    
    # Apply manual percentage-based cropping
    height, width = image_np.shape
    crop_y = int(height * crop_percentage)
    crop_x = int(width * crop_percentage)
    
    # Ensure we don't crop the entire image
    if crop_y * 2 >= height or crop_x * 2 >= width:
        crop_y = max(1, int(height * 0.05))
        crop_x = max(1, int(width * 0.05))
    
    # Perform the crop
    image_np = image_np[crop_y:height-crop_y, crop_x:width-crop_x]
    
    # Apply Otsu thresholding
    _, image_binarized = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Morphological Opening to remove small noise dots
    kernel = np.ones((2, 2), np.uint8)
    image_binarized = cv2.morphologyEx(image_binarized, cv2.MORPH_CLOSE, kernel)

    # Convert to PIL Image and pad to target size
    image_pil = Image.fromarray(image_binarized)
    image_padded = ImageOps.pad(image_pil, target_size, method=Image.Resampling.NEAREST, color=255)

    # Convert to RGB with black text on white background
    image_rgb = ImageOps.colorize(image_padded, black="black", white="white")

    return image_rgb
