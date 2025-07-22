import cv2
import numpy as np
from typing import List, Dict, Optional

class SimpleFaceDetector:
    """
    Simple face detector using OpenCV Haar cascades.
    Fallback when MediaPipe is not available.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, max_faces: int = 10):
        self.confidence_threshold = confidence_threshold
        self.max_faces = max_faces
        
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Performance optimization
        self.frame_skip = 2
        self.frame_count = 0
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using Haar cascades.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face dictionaries with bbox, confidence, face_crop, etc.
        """
        self.frame_count += 1
        
        # Resize image for faster processing
        height, width = image.shape[:2]
        if width > 640:
            scale_factor = 640 / width
            new_width = 640
            new_height = int(height * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
        else:
            scale_factor = 1.0
            resized_image = image
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_rect = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for i, (x, y, w, h) in enumerate(faces_rect[:self.max_faces]):
            # Scale back to original size if image was resized
            if scale_factor != 1.0:
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)
            
            # Ensure coordinates are within image bounds
            orig_h, orig_w = image.shape[:2]
            x = max(0, min(x, orig_w - 1))
            y = max(0, min(y, orig_h - 1))
            w = min(w, orig_w - x)
            h = min(h, orig_h - y)
            
            # Extract face crop
            face_crop = image[y:y+h, x:x+w] if h > 0 and w > 0 else np.zeros((1, 1, 3), dtype=np.uint8)
            
            # Calculate confidence (Haar cascades don't provide confidence, so we estimate)
            confidence = 0.8  # Default confidence for Haar cascade detections
            
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'face_crop': face_crop,
                'keypoints': [],  # Not available with Haar cascades
                'center': (x + w // 2, y + h // 2)
            })
        
        return faces