import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightAgeGenderEstimator:
    """
    Lightweight age and gender estimator optimized for CPU deployment.
    Uses MobileNetV2-based models for real-time performance.
    """
    
    def __init__(self, model_path: str = "models/"):
        """
        Initialize the age and gender estimator.
        
        Args:
            model_path: Path to the directory containing models
        """
        self.model_path = model_path
        self.age_model = None
        self.gender_model = None
        self.input_size = (224, 224)
        
        # Age groups for classification
        self.age_groups = [
            "0-10", "11-20", "21-30", "31-40", 
            "41-50", "51-60", "61-70", "71+"
        ]
        
        # Gender labels
        self.gender_labels = ["Male", "Female"]
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the age and gender estimation models."""
        try:
            # Try to load custom trained models first
            age_model_path = os.path.join(self.model_path, "age_model.h5")
            gender_model_path = os.path.join(self.model_path, "gender_model.h5")
            
            if os.path.exists(age_model_path) and os.path.exists(gender_model_path):
                logger.info("Loading custom trained models...")
                self.age_model = tf.keras.models.load_model(age_model_path)
                self.gender_model = tf.keras.models.load_model(gender_model_path)
            else:
                logger.info("Custom models not found. Creating lightweight models...")
                self._create_lightweight_models()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.info("Creating fallback lightweight models...")
            self._create_lightweight_models()
    
    def _create_lightweight_models(self):
        """Create lightweight MobileNetV2-based models for age and gender estimation."""
        # Base model (MobileNetV2 without top layers)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Age estimation model
        age_model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.age_groups), activation='softmax', name='age_output')
        ])
        
        # Gender estimation model
        gender_model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.gender_labels), activation='softmax', name='gender_output')
        ])
        
        # Compile models
        age_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        gender_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.age_model = age_model
        self.gender_model = gender_model
        
        logger.info("Lightweight models created successfully!")
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model inference.
        
        Args:
            face_image: Face crop as numpy array
            
        Returns:
            Preprocessed image ready for model inference
        """
        # Resize to model input size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def estimate_age(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Estimate age group from face image.
        
        Args:
            face_image: Face crop as numpy array
            
        Returns:
            Tuple of (predicted_age_group, confidence)
        """
        if self.age_model is None:
            return "Unknown", 0.0
        
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_image)
            
            # Make prediction
            predictions = self.age_model.predict(processed_face, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            predicted_age_group = self.age_groups[predicted_class]
            
            return predicted_age_group, confidence
            
        except Exception as e:
            logger.error(f"Error in age estimation: {e}")
            return "Unknown", 0.0
    
    def estimate_gender(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Estimate gender from face image.
        
        Args:
            face_image: Face crop as numpy array
            
        Returns:
            Tuple of (predicted_gender, confidence)
        """
        if self.gender_model is None:
            return "Unknown", 0.0
        
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_image)
            
            # Make prediction
            predictions = self.gender_model.predict(processed_face, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            predicted_gender = self.gender_labels[predicted_class]
            
            return predicted_gender, confidence
            
        except Exception as e:
            logger.error(f"Error in gender estimation: {e}")
            return "Unknown", 0.0
    
    def estimate_age_gender(self, face_image: np.ndarray) -> Dict:
        """
        Estimate both age and gender from face image.
        
        Args:
            face_image: Face crop as numpy array
            
        Returns:
            Dictionary containing age and gender predictions
        """
        age_group, age_confidence = self.estimate_age(face_image)
        gender, gender_confidence = self.estimate_gender(face_image)
        
        return {
            'age_group': age_group,
            'age_confidence': age_confidence,
            'gender': gender,
            'gender_confidence': gender_confidence
        }
    
    def batch_estimate(self, face_images: List[np.ndarray]) -> List[Dict]:
        """
        Estimate age and gender for multiple faces.
        
        Args:
            face_images: List of face crops
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for face_image in face_images:
            result = self.estimate_age_gender(face_image)
            results.append(result)
        
        return results
    
    def get_age_numeric_estimate(self, age_group: str) -> int:
        """
        Convert age group to numeric estimate (middle of range).
        
        Args:
            age_group: Age group string (e.g., "21-30")
            
        Returns:
            Numeric age estimate
        """
        age_mapping = {
            "0-10": 5,
            "11-20": 15,
            "21-30": 25,
            "31-40": 35,
            "41-50": 45,
            "51-60": 55,
            "61-70": 65,
            "71+": 75
        }
        
        return age_mapping.get(age_group, 30)  # Default to 30 if unknown
    
    def save_models(self, save_path: str = "models/"):
        """
        Save the trained models.
        
        Args:
            save_path: Directory to save models
        """
        os.makedirs(save_path, exist_ok=True)
        
        if self.age_model:
            age_model_path = os.path.join(save_path, "age_model.h5")
            self.age_model.save(age_model_path)
            logger.info(f"Age model saved to {age_model_path}")
        
        if self.gender_model:
            gender_model_path = os.path.join(save_path, "gender_model.h5")
            self.gender_model.save(gender_model_path)
            logger.info(f"Gender model saved to {gender_model_path}")


class SimpleDemographicPredictor:
    """
    Simplified demographic predictor using heuristics when ML models are not available.
    This serves as a fallback for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the simple predictor."""
        self.age_groups = [
            "0-10", "11-20", "21-30", "31-40", 
            "41-50", "51-60", "61-70", "71+"
        ]
        self.gender_labels = ["Male", "Female"]
    
    def estimate_age_gender(self, face_image: np.ndarray) -> Dict:
        """
        Simple heuristic-based estimation (for demo purposes).
        In practice, this would use actual computer vision techniques.
        
        Args:
            face_image: Face crop as numpy array
            
        Returns:
            Dictionary containing age and gender predictions
        """
        # This is a simplified placeholder - in real implementation,
        # you would use actual computer vision techniques or pre-trained models
        
        # Random predictions for demonstration
        import random
        
        age_group = random.choice(self.age_groups)
        gender = random.choice(self.gender_labels)
        
        return {
            'age_group': age_group,
            'age_confidence': random.uniform(0.7, 0.95),
            'gender': gender,
            'gender_confidence': random.uniform(0.7, 0.95)
        }