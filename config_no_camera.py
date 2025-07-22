# Modified config for systems without camera
import os

# Use video file instead of camera
CAMERA_INDEX = "test_video.avi"  # Use test video file
# CAMERA_INDEX = 0  # Uncomment when camera is available

# Rest of the configuration
FACE_DETECTION_CONFIDENCE = 0.6
MAX_FACES = 4
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
TARGET_FPS = 30

USE_IMPROVED_MODELS = True
BATCH_PROCESSING = True
BATCH_SIZE = 4
MODEL_WARM_UP = True

# ... (other settings remain the same)
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
SOCKETIO_ASYNC_MODE = 'threading'

# Age groups and other settings
AGE_GROUPS = [
    "0-2", "3-5", "6-8", "9-12", "13-15", "16-18", "19-22", "23-25",
    "26-28", "29-32", "33-35", "36-38", "39-42", "43-45", "46-48",
    "49-52", "53-55", "56-58", "59-62", "63-65", "66-68", "69-72",
    "73-75", "76-78", "79-82", "83+"
]

GENDER_LABELS = ["Male", "Female"]
