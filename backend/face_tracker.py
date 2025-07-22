import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Face tracking system to maintain consistent face IDs across frames
    and prevent multiple counting of the same person.
    """
    
    def __init__(self, max_face_distance: float = 100.0, max_inactive_frames: int = 30):
        """
        Initialize the face tracker.
        
        Args:
            max_face_distance: Maximum distance to consider faces as the same person
            max_inactive_frames: Number of frames before removing inactive faces
        """
        self.max_face_distance = max_face_distance
        self.max_inactive_frames = max_inactive_frames
        
        # Track faces across frames
        self.tracked_faces = {}  # face_id -> face_info
        self.next_face_id = 1
        self.frame_count = 0
        
        # Statistics tracking (only count unique faces)
        self.unique_faces_seen = set()
        self.face_demographics = {}  # face_id -> demographics
        
    def update_tracks(self, current_faces: List[Dict]) -> List[Dict]:
        """
        Update face tracks with current frame detections.
        
        Args:
            current_faces: List of face detections from current frame
            
        Returns:
            List of tracked faces with consistent IDs
        """
        self.frame_count += 1
        
        # If no current faces, just update inactive frames
        if not current_faces:
            self._update_inactive_faces()
            return []
        
        # Match current faces with tracked faces
        tracked_faces_list = []
        used_track_ids = set()
        
        for face in current_faces:
            face_center = face['center']
            best_match_id = None
            best_distance = float('inf')
            
            # Find the closest tracked face
            for track_id, tracked_face in self.tracked_faces.items():
                if track_id in used_track_ids:
                    continue
                    
                distance = self._calculate_distance(face_center, tracked_face['center'])
                
                if distance < self.max_face_distance and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id
            
            # Assign face ID
            if best_match_id is not None:
                # Update existing track
                face_id = best_match_id
                used_track_ids.add(face_id)
                self.tracked_faces[face_id].update({
                    'center': face_center,
                    'bbox': face['bbox'],
                    'last_seen': self.frame_count,
                    'confidence': face['confidence']
                })
            else:
                # Create new track
                face_id = self.next_face_id
                self.next_face_id += 1
                
                self.tracked_faces[face_id] = {
                    'center': face_center,
                    'bbox': face['bbox'],
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'confidence': face['confidence'],
                    'demographics_updated': False
                }
                
                # Add to unique faces seen
                self.unique_faces_seen.add(face_id)
            
            # Add to result with consistent ID
            tracked_face = face.copy()
            tracked_face['id'] = face_id
            # Only mark as new if this face hasn't been processed for demographics yet
            tracked_face['is_new'] = not self.tracked_faces[face_id].get('demographics_updated', False)
            tracked_faces_list.append(tracked_face)
        
        # Remove inactive faces
        self._remove_inactive_faces()
        
        return tracked_faces_list
    
    def update_face_demographics(self, face_id: int, demographics: Dict) -> bool:
        """
        Update demographics for a tracked face.
        
        Args:
            face_id: ID of the face
            demographics: Demographic information
            
        Returns:
            True if this is a new unique face (should be counted in statistics)
        """
        if face_id in self.tracked_faces:
            # Check if this face already has demographics
            is_new_unique_face = not self.tracked_faces[face_id].get('demographics_updated', False)
            
            # Update demographics
            self.face_demographics[face_id] = demographics
            self.tracked_faces[face_id]['demographics_updated'] = True
            
            return is_new_unique_face
        
        return False
    
    def get_unique_face_count(self) -> int:
        """Get the total count of unique faces seen."""
        return len(self.unique_faces_seen)
    
    def get_active_face_count(self) -> int:
        """Get the count of currently active (visible) faces."""
        return len([f for f in self.tracked_faces.values() 
                   if self.frame_count - f['last_seen'] <= 5])  # Active in last 5 frames
    
    def get_demographics_summary(self) -> Dict:
        """
        Get demographics summary for all unique faces seen.
        
        Returns:
            Dictionary with gender and age statistics
        """
        gender_counts = defaultdict(int)
        age_counts = defaultdict(int)
        
        for face_id, demographics in self.face_demographics.items():
            gender = demographics.get('gender', 'Unknown')
            age_group = demographics.get('age_group', 'Unknown')
            
            gender_counts[gender] += 1
            age_counts[age_group] += 1
        
        return {
            'faces_by_gender': dict(gender_counts),
            'faces_by_age': dict(age_counts),
            'total_unique_faces': len(self.face_demographics)
        }
    
    def _calculate_distance(self, center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two face centers."""
        x1, y1 = center1
        x2, y2 = center2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def _update_inactive_faces(self):
        """Update last seen frame for all faces (no current detections)."""
        # This method is called when no faces are detected
        pass
    
    def _remove_inactive_faces(self):
        """Remove faces that haven't been seen for too long."""
        inactive_faces = []
        
        for face_id, face_info in self.tracked_faces.items():
            frames_since_seen = self.frame_count - face_info['last_seen']
            if frames_since_seen > self.max_inactive_frames:
                inactive_faces.append(face_id)
        
        # Remove inactive faces
        for face_id in inactive_faces:
            del self.tracked_faces[face_id]
            logger.debug(f"Removed inactive face {face_id}")
    
    def reset(self):
        """Reset the tracker state."""
        self.tracked_faces.clear()
        self.unique_faces_seen.clear()
        self.face_demographics.clear()
        self.next_face_id = 1
        self.frame_count = 0
        logger.info("Face tracker reset")


class OptimizedFaceDetector:
    """
    Optimized face detector with frame skipping and caching for better performance.
    """
    
    def __init__(self, base_detector, frame_skip: int = 2, cache_duration: int = 3):
        """
        Initialize optimized detector.
        
        Args:
            base_detector: Base face detector instance
            frame_skip: Skip every N frames for detection
            cache_duration: Keep detection results for N frames
        """
        self.base_detector = base_detector
        self.frame_skip = frame_skip
        self.cache_duration = cache_duration
        
        self.frame_count = 0
        self.last_detection_frame = -1
        self.cached_faces = []
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces with optimization (frame skipping and caching).
        
        Args:
            image: Input image
            
        Returns:
            List of face detections
        """
        self.frame_count += 1
        
        # Check if we should skip this frame
        if (self.frame_count - self.last_detection_frame) < self.frame_skip:
            # Return cached results if available and recent
            if (self.frame_count - self.last_detection_frame) <= self.cache_duration:
                return self.cached_faces
        
        # Perform actual detection
        faces = self.base_detector.detect_faces(image)
        
        # Update cache
        self.cached_faces = faces
        self.last_detection_frame = self.frame_count
        
        return faces
    
    def reset_cache(self):
        """Reset the detection cache."""
        self.cached_faces = []
        self.last_detection_frame = -1