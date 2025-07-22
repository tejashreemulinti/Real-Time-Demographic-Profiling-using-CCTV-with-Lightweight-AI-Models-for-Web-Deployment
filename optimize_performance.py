#!/usr/bin/env python3
"""
Performance optimization script for Real-Time Demographic Profiling System
This script optimizes system settings for better accuracy and speed.
"""

import os
import sys
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_tensorflow():
    """Optimize TensorFlow settings for better performance."""
    try:
        # Enable mixed precision for faster inference
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled for TensorFlow")
        
        # Configure GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} GPUs")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")
        else:
            logger.info("No GPU detected, using CPU optimization")
            
        # Enable XLA compilation for faster execution
        tf.config.optimizer.set_jit(True)
        logger.info("XLA compilation enabled")
        
        return True
        
    except Exception as e:
        logger.error(f"TensorFlow optimization failed: {e}")
        return False

def optimize_system_settings():
    """Optimize system environment variables."""
    optimizations = {
        'TF_CPP_MIN_LOG_LEVEL': '2',  # Reduce TensorFlow logging
        'TF_ENABLE_ONEDNN_OPTS': '1',  # Enable oneDNN optimizations
        'OMP_NUM_THREADS': '4',  # OpenMP thread count
        'TF_NUM_INTEROP_THREADS': '2',  # TensorFlow inter-op parallelism
        'TF_NUM_INTRAOP_THREADS': '4',  # TensorFlow intra-op parallelism
        'CUDA_VISIBLE_DEVICES': '0',  # Use first GPU if available
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed and optimized."""
    required_packages = [
        'opencv-python',
        'mediapipe',
        'tensorflow',
        'numpy',
        'flask',
        'flask-socketio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def optimize_opencv():
    """Optimize OpenCV settings for better performance."""
    try:
        import cv2
        
        # Check if OpenCV is built with optimization
        build_info = cv2.getBuildInformation()
        
        if 'NEON' in build_info or 'AVX' in build_info or 'SSE' in build_info:
            logger.info("✓ OpenCV built with CPU optimizations")
        else:
            logger.warning("OpenCV may not be optimized for this CPU")
        
        # Set OpenCV thread count
        cv2.setNumThreads(4)
        logger.info("OpenCV thread count set to 4")
        
        return True
        
    except Exception as e:
        logger.error(f"OpenCV optimization failed: {e}")
        return False

def create_optimized_models():
    """Create and save optimized models for faster loading."""
    try:
        from backend.improved_age_gender_estimator import ImprovedAgeGenderEstimator
        
        logger.info("Creating optimized models...")
        estimator = ImprovedAgeGenderEstimator()
        
        # Warm up models
        estimator.warm_up_models()
        
        # Save optimized models
        estimator.save_models()
        
        logger.info("✓ Optimized models created and saved")
        return True
        
    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        return False

def test_performance():
    """Test system performance and provide recommendations."""
    try:
        import time
        import numpy as np
        from backend.face_detector import LightweightFaceDetector
        from backend.improved_age_gender_estimator import ImprovedAgeGenderEstimator
        
        logger.info("Running performance tests...")
        
        # Test face detection performance
        detector = LightweightFaceDetector()
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        for _ in range(10):
            faces = detector.detect_faces(test_image)
        detection_time = (time.time() - start_time) / 10
        
        logger.info(f"Face detection average time: {detection_time*1000:.1f}ms")
        
        # Test demographic estimation performance
        estimator = ImprovedAgeGenderEstimator()
        test_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        start_time = time.time()
        for _ in range(10):
            result = estimator.estimate_age_gender(test_face)
        estimation_time = (time.time() - start_time) / 10
        
        logger.info(f"Demographic estimation average time: {estimation_time*1000:.1f}ms")
        
        # Calculate expected FPS
        total_time = detection_time + estimation_time
        expected_fps = 1.0 / total_time if total_time > 0 else 0
        
        logger.info(f"Expected FPS: {expected_fps:.1f}")
        
        if expected_fps >= 15:
            logger.info("✓ Performance is good for real-time processing")
        elif expected_fps >= 10:
            logger.warning("⚠ Performance is acceptable but may need optimization")
        else:
            logger.error("✗ Performance is too slow for real-time processing")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False

def main():
    """Main optimization function."""
    logger.info("Starting performance optimization...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        return False
    
    # Optimize system settings
    optimize_system_settings()
    
    # Optimize TensorFlow
    optimize_tensorflow()
    
    # Optimize OpenCV
    optimize_opencv()
    
    # Create optimized models
    create_optimized_models()
    
    # Test performance
    test_performance()
    
    logger.info("Performance optimization completed!")
    
    # Provide recommendations
    logger.info("\n=== RECOMMENDATIONS ===")
    logger.info("1. Use a webcam with at least 720p resolution")
    logger.info("2. Ensure good lighting conditions")
    logger.info("3. Keep faces within 2 meters of the camera")
    logger.info("4. Close other CPU-intensive applications")
    logger.info("5. Consider using a dedicated GPU for better performance")
    
    return True

if __name__ == "__main__":
    if main():
        print("\n✓ System optimized successfully!")
        print("You can now run the application with: python app.py")
    else:
        print("\n✗ Optimization failed. Please check the logs.")
        sys.exit(1)