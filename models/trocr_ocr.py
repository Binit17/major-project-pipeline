import os
import re
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import math
import time

class TrOCROCR:
    def __init__(self, device, model_name="microsoft/trocr-large-handwritten", cache_dir=None, 
                 initial_batch_size=8, min_batch_size=1):
        """
        Initialize TrOCR with dynamic batch sizing
        
        Args:
            device (torch.device): Computing device
            model_name (str): Model name/path
            cache_dir (str, optional): Directory to store downloaded models
            initial_batch_size (int): Starting batch size to attempt
            min_batch_size (int): Minimum batch size before falling back to single processing
        """
        # Set default cache directory if not provided
        if cache_dir is None:
            cache_dir = os.path.expanduser("./trocr_models")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download and cache models
        self.processor = TrOCRProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(device)
        
        self.device = device
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.optimal_batch_size = initial_batch_size  # Will be adjusted dynamically
    
    def recognize_text(self, image):
        """
        Perform OCR on a single image
        """
        # Generate initial text
        with torch.no_grad():  # Prevent gradient calculation during inference
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return raw_text
    
    def _process_sub_batch(self, sub_images):
        """Process a sub-batch of images with error handling"""
        try:
            with torch.no_grad():
                pixel_values = self.processor(images=sub_images, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = self.model.generate(pixel_values)
                return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "the launch timed out" in str(e):
                # If batch size is already at minimum, process images one by one
                if len(sub_images) <= self.min_batch_size:
                    return [self.recognize_text(img) for img in sub_images]
                
                # Otherwise, reduce batch size further and try again
                new_batch_size = max(self.min_batch_size, len(sub_images) // 2)
                self.optimal_batch_size = min(self.optimal_batch_size, new_batch_size)  # Update optimal batch size
                
                # Clear GPU cache to recover memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    time.sleep(0.5)  # Short pause to ensure memory is freed
                
                # Process with reduced batch size
                results = []
                for i in range(0, len(sub_images), new_batch_size):
                    sub_batch = sub_images[i:i+new_batch_size]
                    results.extend(self._process_sub_batch(sub_batch))
                return results
            else:
                # For other errors, raise them
                raise
    
    def recognize_batch(self, images):
        """
        Dynamically process images in optimal batch sizes
        
        Args:
            images: List of PIL images to process
        
        Returns:
            List of recognized text strings
        """
        if not images:
            return []
        
        # Clear GPU cache before starting batch processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Determine current batch size based on recent processing history
        current_batch_size = self.optimal_batch_size
        
        # Process in sub-batches
        results = []
        for i in range(0, len(images), current_batch_size):
            # Get the next sub-batch
            sub_batch = images[i:i+current_batch_size]
            
            # Process this sub-batch
            sub_results = self._process_sub_batch(sub_batch)
            results.extend(sub_results)
            
            # If we processed successfully, we might try increasing batch size slightly
            if len(sub_batch) == current_batch_size and i + current_batch_size < len(images):
                # Potentially increase batch size if we've been successful
                if current_batch_size < self.initial_batch_size:
                    # Gradually increase by 25% (rounded up)
                    current_batch_size = min(self.initial_batch_size, 
                                            math.ceil(current_batch_size * 1.25))
                    self.optimal_batch_size = current_batch_size
        
        return results
