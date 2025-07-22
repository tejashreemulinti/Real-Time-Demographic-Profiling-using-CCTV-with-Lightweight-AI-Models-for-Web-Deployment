import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Callable
import logging
from collections import deque

from .face_detector import LightweightFaceDetector
from .age_gender_estimator import LightweightAgeGenderEstimator, SimpleDemographicPredictor
from .improved_age_gender_estimator import ImprovedAgeGenderEstimator, FastDemographicPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Real-time video processor for demographic analysis.
    Handles video stream processing, face detection, and demographic estimation.
    """
    
    def __init__(self, source=0, use_lightweight_models=True, max_faces=4):
        """
        Initialize the video processor.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            use_lightweight_models: Whether to use lightweight ML models
            max_faces: Maximum number of faces to process per frame
        """
        self.source = source
        self.max_faces = max_faces
        self.use_lightweight_models = use_lightweight_models
        
        # Initialize video capture
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        
        # Initialize AI components
        self.face_detector = LightweightFaceDetector(
            confidence_threshold=0.7, 
            max_faces=max_faces
        )
        
        if use_lightweight_models:
            try:
                # Try improved estimator first
                self.demographic_estimator = ImprovedAgeGenderEstimator()
                logger.info("Using improved ML models for demographic estimation")
                # Warm up models for faster first inference
                self.demographic_estimator.warm_up_models()
            except Exception as e:
                logger.warning(f"Failed to load improved models: {e}. Trying fallback.")
                try:
                    self.demographic_estimator = FastDemographicPredictor()
                    logger.info("Using fast demographic predictor")
                except Exception as e2:
                    logger.warning(f"Failed to load fast predictor: {e2}. Using simple predictor.")
                    self.demographic_estimator = SimpleDemographicPredictor()
        else:
            self.demographic_estimator = FastDemographicPredictor()
            logger.info("Using fast demographic predictor")
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Frame processing queue for threading
        self.frame_queue = deque(maxlen=5)
        self.result_queue = deque(maxlen=5)
        
        # Statistics tracking
        self.statistics = {
            'total_faces_detected': 0,
            'faces_by_gender': {'Male': 0, 'Female': 0, 'Unknown': 0},
            'faces_by_age': {
                '0-10': 0, '11-20': 0, '21-30': 0, '31-40': 0,
                '41-50': 0, '51-60': 0, '61-70': 0, '71+': 0, 'Unknown': 0
            },
            'processing_times': deque(maxlen=100),
            'session_start_time': time.time()
        }
        
        # Callbacks for real-time updates
        self.frame_callback = None
        self.statistics_callback = None
        
        # Threading
        self.processing_thread = None
        self.lock = threading.Lock()
    
    def initialize_camera(self) -> bool:
        """
        Initialize the camera/video source.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture test frame")
                return False
            
            logger.info(f"Camera initialized successfully. Frame size: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for demographic analysis.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing detection results and statistics
        """
        start_time = time.time()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Process demographics for each face
        face_demographics = []
        for i, face in enumerate(faces):
            if face['face_crop'].size > 0:
                # Estimate demographics
                demographics = self.demographic_estimator.estimate_age_gender(face['face_crop'])
                
                # Combine face detection and demographic data
                face_data = {
                    'id': i,
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'demographics': demographics,
                    'center': face['center']
                }
                face_demographics.append(face_data)
                
                # Update statistics
                self._update_statistics(demographics)
        
        # Draw annotations on frame
        annotated_frame = self._draw_annotations(frame.copy(), face_demographics)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.statistics['processing_times'].append(processing_time)
        
        # Update FPS
        self._update_fps()
        
        result = {
            'frame': annotated_frame,
            'faces': face_demographics,
            'fps': self.current_fps,
            'processing_time': processing_time,
            'face_count': len(face_demographics),
            'statistics': self.get_statistics()
        }
        
        return result
    
    def _draw_annotations(self, frame: np.ndarray, face_demographics: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and demographic information on the frame.
        
        Args:
            frame: Input frame
            face_demographics: List of face data with demographics
            
        Returns:
            Annotated frame
        """
        for face_data in face_demographics:
            x, y, w, h = face_data['bbox']
            demographics = face_data['demographics']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare text
            age_text = f"Age: {demographics['age_group']} ({demographics['age_confidence']:.2f})"
            gender_text = f"Gender: {demographics['gender']} ({demographics['gender_confidence']:.2f})"
            
            # Draw text background
            text_y = y - 10
            cv2.rectangle(frame, (x, text_y - 35), (x + 250, text_y + 5), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, age_text, (x + 5, text_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, gender_text, (x + 5, text_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw face ID
            cv2.putText(frame, f"#{face_data['id']}", (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw FPS and statistics
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(face_demographics)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _update_statistics(self, demographics: Dict):
        """Update running statistics with new demographic data."""
        with self.lock:
            self.statistics['total_faces_detected'] += 1
            
            # Update gender statistics
            gender = demographics.get('gender', 'Unknown')
            if gender in self.statistics['faces_by_gender']:
                self.statistics['faces_by_gender'][gender] += 1
            
            # Update age statistics
            age_group = demographics.get('age_group', 'Unknown')
            if age_group in self.statistics['faces_by_age']:
                self.statistics['faces_by_age'][age_group] += 1
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        with self.lock:
            stats = self.statistics.copy()
            
            # Calculate additional metrics
            runtime = time.time() - stats['session_start_time']
            avg_processing_time = np.mean(list(stats['processing_times'])) if stats['processing_times'] else 0
            
            # Convert deque to list for JSON serialization
            stats['processing_times'] = list(stats['processing_times'])
            
            stats.update({
                'session_runtime': runtime,
                'average_processing_time': avg_processing_time,
                'current_fps': self.current_fps
            })
            
            return stats
    
    def reset_statistics(self):
        """Reset all statistics."""
        with self.lock:
            self.statistics = {
                'total_faces_detected': 0,
                'faces_by_gender': {'Male': 0, 'Female': 0, 'Unknown': 0},
                'faces_by_age': {
                    '0-10': 0, '11-20': 0, '21-30': 0, '31-40': 0,
                    '41-50': 0, '51-60': 0, '61-70': 0, '71+': 0, 'Unknown': 0
                },
                'processing_times': deque(maxlen=100),
                'session_start_time': time.time()
            }
    
    def start_processing(self, threaded=True):
        """
        Start video processing.
        
        Args:
            threaded: Whether to run processing in a separate thread
        """
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        
        if threaded:
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Started threaded video processing")
        else:
            self._processing_loop()
        
        return True
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Store current frame
            self.current_frame = frame.copy()
            
            # Process frame
            try:
                result = self.process_frame(frame)
                self.processed_frame = result['frame']
                
                # Call callbacks if set
                if self.frame_callback:
                    self.frame_callback(result)
                
                if self.statistics_callback:
                    self.statistics_callback(result['statistics'])
                    
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue
    
    def stop_processing(self):
        """Stop video processing."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        logger.info("Video processing stopped")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current processed frame."""
        return self.processed_frame
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Get the current raw frame."""
        return self.current_frame
    
    def set_callbacks(self, frame_callback: Callable = None, 
                     statistics_callback: Callable = None):
        """
        Set callbacks for real-time updates.
        
        Args:
            frame_callback: Function called with each processed frame result
            statistics_callback: Function called with updated statistics
        """
        self.frame_callback = frame_callback
        self.statistics_callback = statistics_callback
    
    def capture_screenshot(self, filename: str = None) -> str:
        """
        Capture a screenshot of the current processed frame.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved screenshot
        """
        if self.processed_frame is None:
            raise ValueError("No processed frame available")
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.jpg"
        
        cv2.imwrite(filename, self.processed_frame)
        logger.info(f"Screenshot saved: {filename}")
        
        return filename
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_processing()