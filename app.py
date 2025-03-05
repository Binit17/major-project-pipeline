import streamlit as st
import torch
from PIL import Image, ImageDraw
import io
import re
import os

from models.yolo_detector import YOLODetector
from models.resnet_classifier import ResNetClassifier
from models.trocr_ocr import TrOCROCR
from utils.image_processing import preprocess_image, updated_preprocess_image
from utils.box_sorting import sort_boxes

# Define page configuration
st.set_page_config(
    page_title="Strike-Through Text Detection",
    page_icon="📝",
    layout="wide"
)

# Global model cache to prevent repeated loading
@st.cache_resource
def load_models(device, cache_dir):
    """
    Load models once and cache them for subsequent runs
    
    Args:
        device (torch.device): Computing device
        cache_dir (str): Directory to store downloaded models
    
    Returns:
        tuple: Initialized model instances
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load models
    yolo_detector = YOLODetector()
    resnet_classifier = ResNetClassifier(device)
    trocr_ocr = TrOCROCR(device, cache_dir=cache_dir)
    
    return yolo_detector, resnet_classifier, trocr_ocr

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace existing device check with:
st.sidebar.code(f"""
CUDA Available: {torch.cuda.is_available()}
Device Name: {'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}
PyTorch CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
""")

# Define a cache directory for models
MODEL_CACHE_DIR = os.path.expanduser("models/trocr_models")

# Add a title and description
st.title("OCR Pipeline")
st.markdown("""
This application detects and processes text from handwritten text documents:
1. Detects text regions using YOLO
2. Classifies each region as normal text or strike-through
3. Performs OCR on normal text regions
""")

st.sidebar.info(f"Using device: {device}")

class OCRPipeline:
    def __init__(self):
        # Use cached model loading
        with st.sidebar.expander("Model Loading Status", expanded=True):
            yolo_loading = st.empty()
            yolo_loading.text("Loading YOLO detector...")
            
            # Load models using cached resource
            self.yolo_detector, self.resnet_classifier, self.trocr_ocr = load_models(
                device, 
                cache_dir=MODEL_CACHE_DIR
            )
            
            yolo_loading.text("✅ YOLO detector loaded")
            st.sidebar.success("All models loaded successfully!")

    def process_image(self, image):
        """Main OCR processing pipeline"""
        orig_image = image.convert("RGB")
        img_with_boxes = orig_image.copy()

        # Detect text regions with YOLO
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Detecting text regions with YOLO...")
        all_boxes = self.yolo_detector.detect_text(orig_image)
        progress_bar.progress(30)
        
        # Flatten and sort boxes if needed
        sorted_boxes = sort_boxes([box for sublist in all_boxes for box in sublist])
        progress_bar.progress(40)

        final_text = [''] * len(sorted_boxes)   # Placeholder for correct text position mapping 
        classification_results = []
        valid_images = []  # Store images for batch processing
        valid_indices = [] # Track positions of valid text regions
        draw = ImageDraw.Draw(img_with_boxes)
        
        status_text.text("Classifying text regions and performing OCR...")
        total_boxes = len(sorted_boxes)
        
        for i, box in enumerate(sorted_boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_region = orig_image.crop((x1, y1, x2, y2))
            classifier_input = updated_preprocess_image(cropped_region)

            class_id, confidence = self.resnet_classifier.classify(classifier_input)
            classification_results.append((x1, y1, x2, y2, class_id, confidence))

            # THIS WORKS 
            # if class_id == 0:  # If not strike-through text
            #     ocr_text = self.trocr_ocr.recognize_text(cropped_region)
            #     if ocr_text:  # Only append non-empty text
            #         final_text.append(ocr_text)
            
            # # Update progress based on how many boxes we've processed
            # progress_value = 40 + (i / total_boxes) * 50
            # progress_bar.progress(int(progress_value))
            # THIS WORKS
            
            # TRYING BATCH PROCESSING
            if class_id == 0:
                valid_images.append(cropped_region)
                valid_indices.append(i)  # Track position of valid text
                # Defer OCR processing until after classification

        # Batch process all valid images at once
        if valid_images:
            try:
                batch_results = self.trocr_ocr.recognize_batch(valid_images)
                # Map results back to their original positions
                for idx, text in zip(valid_indices, batch_results):
                    if text:  # Only keep non-empty results
                        final_text[idx] = text
            except RuntimeError as e:
                st.error(f"Batch processing failed: {str(e)}")
                return img_with_boxes, ""

        # Filter out empty strings and join with spaces
        cleaned_text = ' '.join([t for t in final_text if t])
            # TRYING BATCH PROCESSING

        # Visualization
        for x1, y1, x2, y2, class_id, conf in classification_results:
            color = "green" if class_id == 0 else "red"
            label = f"{'Text' if class_id == 0 else 'Strike'} ({conf:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1+5, y1-25), label, fill=color)

        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        # full_text = ' '.join(final_text)
        cleaned_text = re.sub(r'(\w)\.(\s*\w)', r'\1\2', cleaned_text)  # Remove inter-word dots
        cleaned_text = re.sub(r'\s+\.\s+', ' ', cleaned_text)        # Remove floating dots
        # cleaned_text = re.sub(r'\b(\w+)\.', r'\1', cleaned_text)     # Remove word-ending dots
        
        return img_with_boxes, cleaned_text

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = OCRPipeline()

# File uploader
st.subheader("Upload an image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    
    # Process button
    if st.button("Process Image"):
        with st.spinner("Processing image..."):
            # Process the image
            annotated_image, extracted_text = st.session_state.pipeline.process_image(image)
            
            # Display results
            with col2:
                st.subheader("Processed Image")
                st.image(annotated_image, use_container_width=True)
            
            st.subheader("Extracted Text")
            st.write(extracted_text)
            
            # Option to download the annotated image
            buf = io.BytesIO()
            annotated_image.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Annotated Image",
                data=byte_im,
                file_name="annotated_image.jpg",
                mime="image/jpeg",
            )

# Add a section for trying sample images
st.sidebar.markdown("---")
st.sidebar.subheader("Try Sample Images")

sample_images = {
    "Sample 1": "test_images/try1.jpg",
    "Sample 2": "test_images/try2.jpg"
}

selected_sample = st.sidebar.selectbox("Select a sample image", list(sample_images.keys()))

if st.sidebar.button("Process Sample"):
    sample_path = sample_images[selected_sample]
    
    # Check if the sample file exists
    if os.path.exists(sample_path):
        image = Image.open(sample_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Image")
            st.image(image, use_container_width=True)
        
        with st.spinner("Processing sample image..."):
            # Process the image
            annotated_image, extracted_text = st.session_state.pipeline.process_image(image)
            
            # Display results
            with col2:
                st.subheader("Processed Image")
                st.image(annotated_image, use_container_width=True)
            
            st.subheader("Extracted Text")
            st.write(extracted_text)
    else:
        st.sidebar.error(f"Sample file {sample_path} not found!")

# Add additional information in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This application identifies text in medical forms and distinguishes between normal text and strike-through text.
- Green boxes: Valid text (OCR performed)
- Red boxes: Strike-through text (ignored for OCR)
""")