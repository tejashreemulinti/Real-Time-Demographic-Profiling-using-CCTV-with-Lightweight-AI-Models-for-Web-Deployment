#!/usr/bin/env python3
"""
Simple camera availability test and virtual camera setup
"""

import cv2
import numpy as np
import os

def test_camera_availability():
    """Test which camera indices are available."""
    print("🎥 Testing camera availability...")
    
    available_cameras = []
    
    # Test indices 0-3
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {i}: Available - Resolution: {frame.shape}")
                available_cameras.append(i)
            else:
                print(f"❌ Camera {i}: Can't capture frames")
            cap.release()
        else:
            print(f"❌ Camera {i}: Not available")
    
    return available_cameras

def create_virtual_camera_file():
    """Create a sample video file for testing."""
    print("\n📹 Creating virtual camera file for testing...")
    
    try:
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test_video.avi', fourcc, 20.0, (640, 480))
        
        for i in range(100):  # 5 seconds at 20 FPS
            # Create a simple animated frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add text
            cv2.putText(frame, f'Test Frame {i+1}', (200, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add moving circle
            center_x = int(320 + 200 * np.sin(i * 0.1))
            center_y = int(240 + 100 * np.cos(i * 0.1))
            cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
            
            out.write(frame)
        
        out.release()
        print("✅ Created test_video.avi for virtual camera testing")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test video: {e}")
        return False

def test_video_file_as_camera():
    """Test using video file as camera input."""
    print("\n🎬 Testing video file as camera input...")
    
    if not os.path.exists('test_video.avi'):
        if not create_virtual_camera_file():
            return False
    
    try:
        cap = cv2.VideoCapture('test_video.avi')
        
        if not cap.isOpened():
            print("❌ Could not open test video file")
            return False
        
        frame_count = 0
        while frame_count < 5:  # Test first 5 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"✅ Frame {frame_count}: {frame.shape}")
        
        cap.release()
        print(f"✅ Successfully processed {frame_count} frames from video file")
        return True
        
    except Exception as e:
        print(f"❌ Video file test failed: {e}")
        return False

def provide_solutions():
    """Provide solutions for camera issues."""
    print("\n" + "="*60)
    print("🔧 SOLUTIONS FOR VIDEO PROCESSING")
    print("="*60)
    
    print("\n1. 📱 **Use Phone Camera as Webcam:**")
    print("   - Download DroidCam (Android) or EpocCam (iOS)")
    print("   - Connect phone to same WiFi")
    print("   - Use phone as wireless webcam")
    
    print("\n2. 🎥 **Use External USB Webcam:**")
    print("   - Connect any USB webcam")
    print("   - Should appear as /dev/video0 or camera index 0")
    
    print("\n3. 📹 **Use Video File Instead:**")
    print("   - Modify config.py: CAMERA_INDEX = 'test_video.avi'")
    print("   - System will process video file instead of live camera")
    
    print("\n4. 🌐 **Use Browser Camera (Alternative):**")
    print("   - Create HTML5 WebRTC version")
    print("   - Process video in browser using JavaScript")
    
    print("\n5. 🔄 **Run with Virtual Camera:**")
    print("   - Linux: sudo modprobe v4l2loopback")
    print("   - Windows: Install OBS Virtual Camera")
    print("   - macOS: Use CamTwist or similar")

def create_camera_fix():
    """Create a modified config that works without camera."""
    print("\n🔧 Creating camera-free configuration...")
    
    config_content = '''# Modified config for systems without camera
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
'''
    
    try:
        with open('config_no_camera.py', 'w') as f:
            f.write(config_content)
        print("✅ Created config_no_camera.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return False

def main():
    print("🔍 CAMERA AVAILABILITY & VIRTUAL CAMERA SETUP")
    print("="*60)
    
    # Test real cameras
    available_cameras = test_camera_availability()
    
    # Test video file option
    video_file_works = test_video_file_as_camera()
    
    # Create configuration
    config_created = create_camera_fix()
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if available_cameras:
        print(f"✅ Real cameras available at indices: {available_cameras}")
        print("   → You can use the normal configuration")
    else:
        print("❌ No real cameras detected")
        
    if video_file_works:
        print("✅ Video file processing works")
        print("   → You can use test_video.avi as camera input")
    
    if config_created:
        print("✅ Created config_no_camera.py")
        
    # Provide solutions
    provide_solutions()
    
    print("\n🚀 **QUICK FIX TO TEST THE SYSTEM:**")
    print("1. Copy config_no_camera.py over config.py")
    print("2. Run: python app.py")
    print("3. Open: http://localhost:5000")
    print("4. Click 'Start Processing' - will use test video!")

if __name__ == "__main__":
    main()