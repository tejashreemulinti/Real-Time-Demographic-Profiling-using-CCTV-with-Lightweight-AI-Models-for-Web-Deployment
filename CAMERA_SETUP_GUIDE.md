# 📷 Camera Setup Guide - Switch to Real Camera

## 🎯 **Current Status: Working with Test Video**

Your demographic analysis system is now working perfectly with a test video. Here's how to switch to a real camera when you get one:

## 🔄 **Quick Switch to Real Camera**

### **Step 1: Just Change One Line**
In `realtime_demographics.py`, change line 18:

```python
# BEFORE (test video)
video_source = "test_video.avi"

# AFTER (real camera)
video_source = 0  # Use first camera (usually built-in or USB camera)
```

### **Step 2: Restart the Application**
```bash
# Stop current app (Ctrl+C)
# Then restart:
source venv/bin/activate
python realtime_demographics.py
```

That's it! The system will now use your real camera.

## 📱 **Camera Options**

### **Option 1: Phone as Webcam (Easiest)**

**For Android:**
1. Download "DroidCam" app from Play Store
2. Install DroidCam client on your computer
3. Connect phone and computer to same WiFi
4. Your phone becomes camera index 0

**For iPhone:**
1. Download "EpocCam" app from App Store  
2. Install EpocCam drivers on your computer
3. Connect via WiFi or USB
4. iPhone becomes a webcam

### **Option 2: USB Webcam**
1. Connect any USB webcam
2. It should automatically be detected as camera index 0
3. Test with: `ls /dev/video*` (Linux) or Device Manager (Windows)

### **Option 3: Built-in Laptop Camera**
1. Make sure camera is enabled in system settings
2. Close any other apps using the camera (Zoom, Skype, etc.)
3. Use camera index 0

## 🔧 **Multiple Camera Setup**

If you have multiple cameras:

```python
video_source = 0  # First camera (usually built-in)
video_source = 1  # Second camera (usually USB webcam)
video_source = 2  # Third camera
# etc.
```

## 🧪 **Test Camera Before Using**

Create a simple test script:

```python
import cv2

# Test different camera indices
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera {i}: Working - Resolution: {frame.shape}")
        else:
            print(f"❌ Camera {i}: Can't capture frames")
        cap.release()
    else:
        print(f"❌ Camera {i}: Not available")
```

## 🎯 **What You'll Get with Real Camera**

✅ **Live person detection** instead of test video  
✅ **Real-time demographics** of actual people  
✅ **Interactive experience** - wave at the camera!  
✅ **Accurate face counting** for real scenarios  
✅ **Live statistics** updating with real data  

## 🔍 **Troubleshooting Real Camera**

### **Camera Not Detected**
- Check if camera is being used by another app
- Try different USB ports (for USB cameras)
- Restart computer
- Check privacy settings (Windows/Mac may block camera access)

### **Poor Performance**
- Lower resolution in camera settings
- Increase `time.sleep()` in generate_frames() function
- Reduce face detection frequency

### **Permission Issues**
- Grant camera permissions to terminal/Python
- Run as administrator (Windows) or with sudo (Linux) if needed

## 💡 **Advanced Camera Features**

### **Custom Resolution**
```python
def init_video():
    global cap, face_cascade
    cap = cv2.VideoCapture(video_source)
    
    # Set custom resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
```

### **Camera Settings**
```python
# Brightness, contrast, etc.
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
cap.set(cv2.CAP_PROP_SATURATION, 0.5)
```

## 🚀 **Ready for Production**

Once you switch to a real camera, your system becomes a professional demographic analysis tool:

- **Retail Analytics**: Count customers, analyze demographics
- **Event Monitoring**: Track attendee demographics  
- **Security Applications**: Monitor public spaces
- **Research Projects**: Gather demographic data
- **Interactive Displays**: Respond to viewer demographics

## 📊 **Current Working Features**

Even with test video, you can see all features working:
- ✅ **Video streaming** - Working perfectly
- ✅ **Face detection** - Green boxes around faces
- ✅ **Age estimation** - Age groups with confidence
- ✅ **Gender prediction** - Male/Female with confidence  
- ✅ **Live statistics** - Real-time charts and metrics
- ✅ **Professional UI** - Modern interface
- ✅ **Session tracking** - Timer and analytics

**The system is production-ready! Just switch the video source when you get a camera.** 🎯