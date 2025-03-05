import os
import re
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TrOCROCR:
    def __init__(self, device, cache_dir=None):
        """
        Initialize TrOCR with local caching option
        
        Args:
            device (torch.device): Computing device
            cache_dir (str, optional): Directory to store downloaded models
        """
        # Set default cache directory if not provided
        if cache_dir is None:
            cache_dir = os.path.expanduser("./trocr_models")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download and cache models
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-large-handwritten",
            cache_dir=cache_dir
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-handwritten",
            cache_dir=cache_dir
        ).to(device)
        
        self.device = device
    
    
    def recognize_text(self, image):
        """
        Perform OCR with multi-stage text normalization
        """
        # Generate initial text
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return raw_text
    
    
    def recognize_batch(self, images):
        """Batch process multiple images at once"""
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

