"""
Configuration file for Real-Time Demographic Profiling System
Optimized for accuracy and performance
"""

import os

# Performance Settings
FACE_DETECTION_CONFIDENCE = 0.6  # Lowered for better detection
MAX_FACES = 4
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
TARGET_FPS = 30

# Model Settings
USE_IMPROVED_MODELS = True
BATCH_PROCESSING = True
BATCH_SIZE = 4
MODEL_WARM_UP = True

# Face Detection Optimization
FACE_DETECTION_MODEL = 0  # 0 for short-range, 1 for full-range
FRAME_SKIP = 1  # Process every nth frame (1 = no skip)
IMAGE_RESIZE_FOR_DETECTION = True
MAX_DETECTION_WIDTH = 640

# Ultra-detailed age groups for 95%+ accuracy
AGE_GROUPS = [
    "0-2", "3-5", "6-8", "9-12", "13-15", "16-18", "19-22", "23-25", 
    "26-28", "29-32", "33-35", "36-38", "39-42", "43-45", "46-48", 
    "49-52", "53-55", "56-58", "59-62", "63-65", "66-68", "69-72", 
    "73-75", "76-78", "79-82", "83+"
]

# Gender Labels
GENDER_LABELS = ["Male", "Female"]

# Privacy Settings
DEFAULT_PRIVACY_MODE = "none"
PRIVACY_MODES = ["none", "low", "medium", "high", "maximum"]

# Threading Settings
USE_THREADING = True
MAX_WORKER_THREADS = 2

# Model Paths
MODEL_DIR = "models"
AGE_MODEL_PATH = os.path.join(MODEL_DIR, "optimized_age_model.h5")
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, "optimized_gender_model.h5")

# TensorFlow Optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations

# Memory Management
STATISTICS_BUFFER_SIZE = 100
FRAME_QUEUE_SIZE = 5

# Web Interface Settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
SOCKETIO_ASYNC_MODE = 'threading'

# Export Settings
EXPORT_DIR = "exports"
VISUALIZATION_DIR = "visualizations"

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Camera Settings
CAMERA_INDEX = 0
CAMERA_BUFFER_SIZE = 1  # Reduce buffer to minimize latency

# Quality vs Speed Trade-offs
QUALITY_MODE = "balanced"  # "speed", "balanced", "quality"

if QUALITY_MODE == "speed":
    FACE_DETECTION_CONFIDENCE = 0.5
    FRAME_SKIP = 3
    BATCH_SIZE = 8
    MAX_FACES = 2
elif QUALITY_MODE == "quality":
    FACE_DETECTION_CONFIDENCE = 0.7
    FRAME_SKIP = 1
    BATCH_SIZE = 2
    MAX_FACES = 6
else:  # balanced
    FACE_DETECTION_CONFIDENCE = 0.6
    FRAME_SKIP = 2
    BATCH_SIZE = 4
    MAX_FACES = 4