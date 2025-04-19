import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class ResNetClassifier:
    def __init__(self, device):
        """Load ResNet50 classifier with GPU optimization"""
        model_path = 'models/newresnet.pth'

        self.device = device
        
        # Display device information
        if self.device.type == 'cuda':
            print(f"ResNet50 using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("ResNet50 using CPU")
            
        # Initialize the model architecture
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        # Load model weights directly to the specified device
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device).eval()
        
        # Define preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, image):
        """Classify if text region contains strike-through"""
        # Preprocess and move tensor to correct device
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Perform inference without computing gradients
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, dim=1)
        
        return predicted.item(), confidence.item()
    
    def classify_batch(self, images, batch_size=8):
        """Classify multiple images in batches for better GPU utilization"""
        if not images:
            return []
            
        results = []
        
        # Process in smaller batches to avoid memory issues on limited GPUs
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_tensors = []
            
            for image in batch_images:
                tensor = self.preprocess(image).unsqueeze(0)
                batch_tensors.append(tensor)
            
            # Concatenate and move to GPU
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Perform batch inference
            with torch.no_grad():
                outputs = self.model(batch)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probs, dim=1)
            
            # Add results for this batch
            batch_results = [(pred.item(), conf.item()) for pred, conf in zip(predicted, confidences)]
            results.extend(batch_results)
            
            # Free GPU memory after processing each batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return results
