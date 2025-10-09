import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceAnonymizer:
    """
    Face anonymization module with various privacy protection methods.
    Supports blurring, pixelation, masking, and face replacement.
    """
    
    def __init__(self):
        """Initialize the face anonymizer."""
        self.anonymization_methods = {
            'blur': self._blur_face,
            'pixelate': self._pixelate_face,
            'mask': self._mask_face,
            'black_bar': self._black_bar_face,
            'emoji': self._emoji_face
        }
    
    def anonymize_faces(self, image: np.ndarray, faces: List[Dict], 
                       method: str = 'blur', intensity: float = 1.0) -> np.ndarray:
        """
        Anonymize detected faces in the image.
        
        Args:
            image: Input image
            faces: List of face detection results
            method: Anonymization method ('blur', 'pixelate', 'mask', 'black_bar', 'emoji')
            intensity: Anonymization intensity (0.0 to 1.0)
            
        Returns:
            Image with anonymized faces
        """
        if method not in self.anonymization_methods:
            logger.warning(f"Unknown anonymization method: {method}. Using blur.")
            method = 'blur'
        
        anonymized_image = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                # Apply anonymization method
                anonymized_image = self.anonymization_methods[method](
                    anonymized_image, (x, y, w, h), intensity
                )
        
        return anonymized_image
    
    def _blur_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                   intensity: float) -> np.ndarray:
        """
        Apply Gaussian blur to face region.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            intensity: Blur intensity (0.0 to 1.0)
            
        Returns:
            Image with blurred face
        """
        x, y, w, h = bbox
        
        # Calculate kernel size based on intensity
        kernel_size = int(max(15, min(w, h) * intensity * 0.3))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Extract face region
        face_region = image[y:y+h, x:x+w].copy()
        
        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
        
        # Replace face region with blurred version
        image[y:y+h, x:x+w] = blurred_face
        
        return image
    
    def _pixelate_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                      intensity: float) -> np.ndarray:
        """
        Apply pixelation to face region.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            intensity: Pixelation intensity (0.0 to 1.0)
            
        Returns:
            Image with pixelated face
        """
        x, y, w, h = bbox
        
        # Calculate pixel size based on intensity
        pixel_size = max(4, int(min(w, h) * (1.0 - intensity) * 0.1))
        
        # Extract face region
        face_region = image[y:y+h, x:x+w].copy()
        
        # Resize down and up to create pixelation effect
        small_w = max(1, w // pixel_size)
        small_h = max(1, h // pixel_size)
        
        # Resize down
        small_face = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize back up with nearest neighbor interpolation for pixelated effect
        pixelated_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Replace face region
        image[y:y+h, x:x+w] = pixelated_face
        
        return image
    
    def _mask_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                   intensity: float) -> np.ndarray:
        """
        Apply a solid color mask to face region.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            intensity: Mask opacity (0.0 to 1.0)
            
        Returns:
            Image with masked face
        """
        x, y, w, h = bbox
        
        # Create mask color (dark gray)
        mask_color = (64, 64, 64)
        
        # Create overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), mask_color, -1)
        
        # Blend with original image based on intensity
        alpha = intensity
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        return image
    
    def _black_bar_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                       intensity: float) -> np.ndarray:
        """
        Apply black bar censorship to face region.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            intensity: Bar height ratio (0.0 to 1.0)
            
        Returns:
            Image with black bar over eyes
        """
        x, y, w, h = bbox
        
        # Calculate bar position (over eyes area)
        bar_height = int(h * 0.3 * intensity)
        bar_y = y + int(h * 0.3)  # Position over eyes
        
        # Draw black rectangle
        cv2.rectangle(image, (x, bar_y), (x + w, bar_y + bar_height), (0, 0, 0), -1)
        
        return image
    
    def _emoji_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                   intensity: float) -> np.ndarray:
        """
        Replace face with a simple emoji-like representation.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            intensity: Emoji size ratio (0.0 to 1.0)
            
        Returns:
            Image with emoji face
        """
        x, y, w, h = bbox
        
        # Create a simple smiley face
        center_x = x + w // 2
        center_y = y + h // 2
        radius = int(min(w, h) * 0.4 * intensity)
        
        # Draw face circle (yellow background)
        cv2.circle(image, (center_x, center_y), radius, (0, 255, 255), -1)
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), 2)
        
        # Draw eyes
        eye_radius = max(2, radius // 8)
        eye_offset_x = radius // 3
        eye_offset_y = radius // 4
        
        cv2.circle(image, (center_x - eye_offset_x, center_y - eye_offset_y), 
                  eye_radius, (0, 0, 0), -1)
        cv2.circle(image, (center_x + eye_offset_x, center_y - eye_offset_y), 
                  eye_radius, (0, 0, 0), -1)
        
        # Draw smile
        smile_start_x = center_x - radius // 2
        smile_end_x = center_x + radius // 2
        smile_y = center_y + radius // 3
        smile_radius = radius // 2
        
        cv2.ellipse(image, (center_x, smile_y), (smile_radius, smile_radius // 2), 
                   0, 0, 180, (0, 0, 0), 2)
        
        return image
    
    def create_privacy_zones(self, image: np.ndarray, zones: List[Tuple[int, int, int, int]], 
                           method: str = 'blur') -> np.ndarray:
        """
        Apply anonymization to custom privacy zones.
        
        Args:
            image: Input image
            zones: List of bounding boxes for privacy zones
            method: Anonymization method
            
        Returns:
            Image with anonymized privacy zones
        """
        anonymized_image = image.copy()
        
        for zone in zones:
            fake_face = {'bbox': zone}
            anonymized_image = self.anonymize_faces(
                anonymized_image, [fake_face], method, intensity=1.0
            )
        
        return anonymized_image
    
    def selective_anonymization(self, image: np.ndarray, faces: List[Dict], 
                              demographic_filter: Dict, method: str = 'blur') -> np.ndarray:
        """
        Anonymize faces based on demographic criteria.
        
        Args:
            image: Input image
            faces: List of face detection results with demographics
            demographic_filter: Filter criteria (e.g., {'age_group': ['0-10', '11-20']})
            method: Anonymization method
            
        Returns:
            Image with selectively anonymized faces
        """
        faces_to_anonymize = []
        
        for face in faces:
            demographics = face.get('demographics', {})
            should_anonymize = False
            
            # Check age filter
            if 'age_group' in demographic_filter:
                if demographics.get('age_group') in demographic_filter['age_group']:
                    should_anonymize = True
            
            # Check gender filter
            if 'gender' in demographic_filter:
                if demographics.get('gender') in demographic_filter['gender']:
                    should_anonymize = True
            
            # Check confidence thresholds
            if 'min_confidence' in demographic_filter:
                age_conf = demographics.get('age_confidence', 0)
                gender_conf = demographics.get('gender_confidence', 0)
                if min(age_conf, gender_conf) >= demographic_filter['min_confidence']:
                    should_anonymize = True
            
            if should_anonymize:
                faces_to_anonymize.append(face)
        
        return self.anonymize_faces(image, faces_to_anonymize, method)
    
    def get_anonymization_preview(self, image: np.ndarray, faces: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Generate preview images with different anonymization methods.
        
        Args:
            image: Input image
            faces: List of face detection results
            
        Returns:
            Dictionary of method names to preview images
        """
        previews = {}
        
        for method in self.anonymization_methods.keys():
            preview_image = self.anonymize_faces(image.copy(), faces, method, intensity=0.8)
            previews[method] = preview_image
        
        return previews
    
    def batch_anonymize(self, images: List[np.ndarray], face_lists: List[List[Dict]], 
                       method: str = 'blur') -> List[np.ndarray]:
        """
        Anonymize multiple images in batch.
        
        Args:
            images: List of input images
            face_lists: List of face detection results for each image
            method: Anonymization method
            
        Returns:
            List of anonymized images
        """
        anonymized_images = []
        
        for image, faces in zip(images, face_lists):
            anonymized_image = self.anonymize_faces(image, faces, method)
            anonymized_images.append(anonymized_image)
        
        return anonymized_images


class PrivacyModeManager:
    """
    Manager for different privacy modes and settings.
    """
    
    def __init__(self):
        """Initialize privacy mode manager."""
        self.anonymizer = FaceAnonymizer()
        self.privacy_modes = {
            'none': {'enabled': False},
            'low': {
                'enabled': True,
                'method': 'blur',
                'intensity': 0.3,
                'selective': False
            },
            'medium': {
                'enabled': True,
                'method': 'blur',
                'intensity': 0.6,
                'selective': True,
                'filter': {'age_group': ['0-10', '11-20']}  # Protect minors
            },
            'high': {
                'enabled': True,
                'method': 'mask',
                'intensity': 0.8,
                'selective': False
            },
            'maximum': {
                'enabled': True,
                'method': 'black_bar',
                'intensity': 1.0,
                'selective': False
            }
        }
        
        self.current_mode = 'none'
    
    def set_privacy_mode(self, mode: str):
        """
        Set the current privacy mode.
        
        Args:
            mode: Privacy mode name
        """
        if mode in self.privacy_modes:
            self.current_mode = mode
            logger.info(f"Privacy mode set to: {mode}")
        else:
            logger.warning(f"Unknown privacy mode: {mode}")
    
    def apply_privacy_mode(self, image: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Apply the current privacy mode to the image.
        
        Args:
            image: Input image
            faces: List of face detection results
            
        Returns:
            Image with privacy mode applied
        """
        mode_config = self.privacy_modes[self.current_mode]
        
        if not mode_config['enabled']:
            return image
        
        method = mode_config['method']
        intensity = mode_config['intensity']
        
        if mode_config.get('selective', False):
            demographic_filter = mode_config.get('filter', {})
            return self.anonymizer.selective_anonymization(
                image, faces, demographic_filter, method
            )
        else:
            return self.anonymizer.anonymize_faces(image, faces, method, intensity)
    
    def get_current_mode(self) -> str:
        """Get the current privacy mode."""
        return self.current_mode
    
    def get_available_modes(self) -> List[str]:
        """Get list of available privacy modes."""
        return list(self.privacy_modes.keys())