import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict, Optional


class LightweightFaceDetector:
    """
    Lightweight face detector using MediaPipe for real-time performance.
    Optimized for CPU deployment and multi-face detection.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, max_faces: int = 10):
        """
        Initialize the face detector.
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            max_faces: Maximum number of faces to detect
        """
        self.confidence_threshold = confidence_threshold
        self.max_faces = max_faces
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (within 2 meters), 1 for full-range
            min_detection_confidence=confidence_threshold
        )
        
        # Performance optimization
        self.frame_skip = 2  # Process every 2nd frame for speed
        self.frame_count = 0
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in the input image with performance optimization.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries containing face information
        """
        self.frame_count += 1
        
        # Resize image for faster processing
        height, width = image.shape[:2]
        if width > 640:
            scale_factor = 640 / width
            new_width = 640
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height))
        else:
            scale_factor = 1.0
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections[:self.max_faces]:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Scale back to original size if image was resized
                if scale_factor != 1.0:
                    x = int(x / scale_factor)
                    y = int(y / scale_factor)
                    width = int(width / scale_factor)
                    height = int(height / scale_factor)
                
                # Ensure coordinates are within original image bounds
                orig_h, orig_w = image.shape[:2] if scale_factor == 1.0 else (int(h / scale_factor), int(w / scale_factor))
                x = max(0, x)
                y = max(0, y)
                width = min(width, orig_w - x)
                height = min(height, orig_h - y)
                
                # Extract face region from original or resized image
                if scale_factor != 1.0:
                    # Use original image for face extraction
                    face_crop = image[y:y+height, x:x+width] if y+height <= image.shape[0] and x+width <= image.shape[1] else image[y:y+1, x:x+1]
                else:
                    face_crop = image[y:y+height, x:x+width]
                
                # Get confidence score
                confidence = detection.score[0] if detection.score else 0.0
                
                # Get key points (if available)
                keypoints = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        keypoints.append((kp_x, kp_y))
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': confidence,
                    'face_crop': face_crop,
                    'keypoints': keypoints,
                    'center': (x + width // 2, y + height // 2)
                })
        
        return faces
    
    def draw_detections(self, image: np.ndarray, faces: List[Dict], 
                       draw_keypoints: bool = True) -> np.ndarray:
        """
        Draw face detections on the image.
        
        Args:
            image: Input image
            faces: List of face detections
            draw_keypoints: Whether to draw facial keypoints
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for i, face in enumerate(faces):
            x, y, width, height = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + width, y + height), 
                         (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face {i+1}: {confidence:.2f}"
            cv2.putText(result_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints if available
            if draw_keypoints and face['keypoints']:
                for kp_x, kp_y in face['keypoints']:
                    cv2.circle(result_image, (kp_x, kp_y), 3, (255, 0, 0), -1)
        
        return result_image
    
    def get_face_crops(self, image: np.ndarray, faces: List[Dict], 
                      target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """
        Extract and resize face crops for further processing.
        
        Args:
            image: Input image
            faces: List of face detections
            target_size: Target size for face crops
            
        Returns:
            List of resized face crops
        """
        face_crops = []
        
        for face in faces:
            if face['face_crop'].size > 0:
                # Resize face crop
                resized_crop = cv2.resize(face['face_crop'], target_size)
                face_crops.append(resized_crop)
        
        return face_crops
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()