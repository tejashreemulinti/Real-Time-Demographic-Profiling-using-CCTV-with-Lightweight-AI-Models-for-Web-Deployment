#!/usr/bin/env python3
"""
Debug script to test camera initialization and video processing
"""

import cv2
import sys
import os
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_camera_direct():
    """Test camera access directly with OpenCV."""
    print("🎥 Testing direct camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not accessible at index 0")
            return False
        
        print("✅ Camera opened successfully")
        
        # Try to capture a frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            cap.release()
            return False
        
        print(f"✅ Frame captured successfully - Shape: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_video_processor():
    """Test VideoProcessor initialization."""
    print("\n🔧 Testing VideoProcessor initialization...")
    
    try:
        from backend.video_processor import VideoProcessor
        
        processor = VideoProcessor(source=0, use_lightweight_models=True, max_faces=4)
        print("✅ VideoProcessor created successfully")
        
        # Test camera initialization
        if processor.initialize_camera():
            print("✅ Camera initialized in VideoProcessor")
            
            # Test processing a frame
            if processor.cap:
                ret, frame = processor.cap.read()
                if ret:
                    print("✅ Frame capture test successful")
                    
                    # Try processing the frame
                    try:
                        result = processor.process_frame(frame)
                        print(f"✅ Frame processing successful - {len(result['faces'])} faces detected")
                    except Exception as e:
                        print(f"⚠️ Frame processing error: {e}")
                        
                processor.cap.release()
            else:
                print("❌ Camera not initialized properly")
                
        else:
            print("❌ Camera initialization failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ VideoProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_start_processing():
    """Test the start_processing method."""
    print("\n🚀 Testing start_processing method...")
    
    try:
        from backend.video_processor import VideoProcessor
        
        processor = VideoProcessor(source=0, use_lightweight_models=True, max_faces=4)
        
        # Test starting processing
        success = processor.start_processing(threaded=False)
        
        if success:
            print("✅ start_processing returned True")
            
            # Let it run for a short time
            import time
            time.sleep(2)
            
            # Check if it's running
            if processor.is_running:
                print("✅ Processing is running")
            else:
                print("❌ Processing stopped unexpectedly")
                
            # Stop processing
            processor.stop_processing()
            print("✅ Processing stopped successfully")
            
        else:
            print("❌ start_processing returned False")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ start_processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 Camera and Video Processing Debug")
    print("=" * 50)
    
    # Configure logging to see more details
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Direct camera access
    camera_ok = test_camera_direct()
    
    # Test 2: VideoProcessor initialization  
    processor_ok = test_video_processor()
    
    # Test 3: Start processing method
    start_ok = test_start_processing()
    
    print("\n" + "=" * 50)
    print("🏁 SUMMARY")
    print("=" * 50)
    print(f"Camera Access: {'✅ OK' if camera_ok else '❌ FAILED'}")
    print(f"VideoProcessor: {'✅ OK' if processor_ok else '❌ FAILED'}")
    print(f"Start Processing: {'✅ OK' if start_ok else '❌ FAILED'}")
    
    if camera_ok and processor_ok and start_ok:
        print("\n🎉 All tests passed! Video processing should work.")
        print("\n💡 If the web app still doesn't work, the issue might be:")
        print("   - Browser permissions for camera access")
        print("   - WebSocket connection issues")  
        print("   - JavaScript errors in the console")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()