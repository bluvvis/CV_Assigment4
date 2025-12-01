"""
Object Detector - Object recognition on frames
Stage 6: Object Detection (integrated into rendering)
"""

import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class ObjectDetector:
    """Object detection using YOLOv8"""
    
    def __init__(self, config: Dict):
        """
        Initialize object detector
        
        Args:
            config: Configuration from config.yaml (object_detection section)
        """
        self.config = config
        self.model_path = config.get('model', 'yolov8n.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        self.allowed_classes = config.get('allowed_classes', [])
        self.visualize = config.get('visualize', True)
        
        print(f"Loading YOLO model: {self.model_path}")
        # Use GPU if available (for RTX 4070 Super)
        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        self.model = YOLO(self.model_path)
        if device == 'cuda':
            print(f"YOLO model loaded successfully - using GPU acceleration")
        else:
            print("YOLO model loaded successfully - using CPU")
    
    def detect(self, image_path: str) -> Tuple[List[Dict], Optional[Image.Image]]:
        """
        Detect objects in image
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple[List[Dict], Optional[Image.Image]]: 
                - List of detections with bbox, confidence, class
                - Visualized image (if visualize=True)
        """
        # Load image
        image = Image.open(image_path)
        
        # YOLO detection
        results = self.model(image_path, conf=self.confidence_threshold)
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Extract data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Filter by allowed classes
                if self.allowed_classes and class_name not in self.allowed_classes:
                    continue
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class': class_name,
                    'class_id': class_id
                })
        
        # Visualization
        visualized_image = None
        if self.visualize and detections:
            visualized_image = self._draw_detections(image.copy(), detections)
        
        return detections, visualized_image
    
    def _draw_detections(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """
        Draw bounding boxes on image
        
        Args:
            image: PIL Image
            detections: List of detections
            
        Returns:
            Image.Image: Image with drawn bounding boxes
        """
        draw = ImageDraw.Draw(image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class']
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            # Text with class and confidence
            label = f"{class_name} {confidence:.2f}"
            
            # Text size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Background for text
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill='red',
                outline='red'
            )
            
            # Text
            draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
        
        return image
    
    def save_detections(self, detections: List[Dict], output_path: str):
        """Save detections to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({'detections': detections}, f, indent=2)
        
        print(f"Saved {len(detections)} detections to {output_path}")


if __name__ == "__main__":
    # Testing (requires image)
    import sys
    
    if len(sys.argv) > 1:
        config = {
            'model': 'yolov8n.pt',
            'confidence_threshold': 0.3,
            'allowed_classes': ['person', 'chair', 'table'],
            'visualize': True
        }
        
        detector = ObjectDetector(config)
        detections, vis_image = detector.detect(sys.argv[1])
        
        print(f"Detected {len(detections)} objects:")
        for det in detections:
            print(f"  {det['class']}: {det['confidence']:.2f}")
        
        if vis_image:
            vis_image.save("detection_result.jpg")
            print("Saved visualization to detection_result.jpg")
    else:
        print("Usage: python object_detector.py <image_path>")

