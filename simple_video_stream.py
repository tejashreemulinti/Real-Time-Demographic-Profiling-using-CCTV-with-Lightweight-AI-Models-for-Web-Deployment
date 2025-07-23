#!/usr/bin/env python3
"""
Minimal video streaming app to fix the streaming issue
"""

from flask import Flask, Response, render_template_string
import cv2
import threading
import time

app = Flask(__name__)

# Global variables
video_source = "test_video.avi"
streaming = False
current_frame = None
cap = None

def init_video():
    """Initialize video capture."""
    global cap
    try:
        cap = cv2.VideoCapture(video_source)
        if cap.isOpened():
            print(f"✅ Video opened: {video_source}")
            return True
        else:
            print(f"❌ Failed to open: {video_source}")
            return False
    except Exception as e:
        print(f"❌ Video init error: {e}")
        return False

def generate_frames():
    """Generate video frames for streaming."""
    global cap, current_frame, streaming
    
    if not cap or not cap.isOpened():
        print("❌ Video not initialized")
        return
    
    frame_count = 0
    while streaming:
        ret, frame = cap.read()
        
        if not ret:
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            
        if ret:
            frame_count += 1
            current_frame = frame.copy()
            
            # Add frame counter
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Video Streaming Active', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.05)  # 20 FPS

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Video Stream Test</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1000px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f0f0f0;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            border: 3px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            padding: 15px 30px;
            margin: 10px;
            font-size: 18px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .start-btn {
            background: #28a745;
            color: white;
        }
        .stop-btn {
            background: #dc3545;
            color: white;
        }
        .status {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            font-weight: bold;
        }
        .status.ready { background: #d4edda; color: #155724; }
        .status.streaming { background: #cce7ff; color: #004085; }
        .status.stopped { background: #f8d7da; color: #721c24; }
        #videoStream {
            max-width: 100%;
            height: auto;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Video Stream Test</h1>
        
        <div id="status" class="status ready">
            Ready to start video streaming
        </div>
        
        <div class="video-container">
            <img id="videoStream" src="/video_feed" alt="Video Stream">
        </div>
        
        <div class="controls">
            <button class="start-btn" onclick="startStream()">
                🟢 Start Video Stream
            </button>
            <button class="stop-btn" onclick="stopStream()">
                🔴 Stop Video Stream
            </button>
            <button onclick="refreshStream()">
                🔄 Refresh Stream
            </button>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 5px;">
            <h3>📋 Instructions:</h3>
            <ol>
                <li>Click "Start Video Stream"</li>
                <li>You should see animated test video with frame counter</li>
                <li>Video will loop automatically</li>
                <li>Frame counter should increment continuously</li>
            </ol>
            
            <h3>🔍 Troubleshooting:</h3>
            <ul>
                <li>If video doesn't start: Check browser console (F12)</li>
                <li>If image shows "broken": Check server terminal for errors</li>
                <li>Try refreshing the page or clicking "Refresh Stream"</li>
            </ul>
        </div>
    </div>
    
    <script>
        let streaming = false;
        
        function startStream() {
            console.log('Starting video stream...');
            
            fetch('/start_stream', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                console.log('Start response:', data);
                if (data.success) {
                    streaming = true;
                    updateStatus('🟢 Video streaming active', 'streaming');
                    
                    // Force refresh the video feed
                    const video = document.getElementById('videoStream');
                    video.src = '/video_feed?' + new Date().getTime();
                } else {
                    updateStatus('❌ Failed to start: ' + data.error, 'stopped');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                updateStatus('❌ Error starting stream', 'stopped');
            });
        }
        
        function stopStream() {
            console.log('Stopping video stream...');
            
            fetch('/stop_stream', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                console.log('Stop response:', data);
                streaming = false;
                updateStatus('🔴 Video stream stopped', 'stopped');
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
        
        function refreshStream() {
            console.log('Refreshing stream...');
            const video = document.getElementById('videoStream');
            video.src = '/video_feed?' + new Date().getTime();
        }
        
        function updateStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
        }
        
        // Auto-refresh every 10 seconds if streaming
        setInterval(() => {
            if (streaming) {
                refreshStream();
            }
        }, 10000);
    </script>
</body>
</html>
    ''')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start video streaming."""
    global streaming
    
    try:
        if not init_video():
            return {'success': False, 'error': 'Failed to initialize video'}
        
        streaming = True
        print("✅ Video streaming started")
        
        return {'success': True, 'message': 'Streaming started'}
        
    except Exception as e:
        print(f"❌ Start stream error: {e}")
        return {'success': False, 'error': str(e)}

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop video streaming."""
    global streaming, cap
    
    streaming = False
    
    if cap:
        cap.release()
        cap = None
    
    print("🔴 Video streaming stopped")
    return {'success': True, 'message': 'Streaming stopped'}

@app.route('/video_feed')
def video_feed():
    """Video streaming route - this is the key function."""
    global streaming
    
    print(f"📹 Video feed requested. Streaming: {streaming}")
    
    if streaming and cap and cap.isOpened():
        try:
            return Response(
                generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
        except Exception as e:
            print(f"❌ Video feed error: {e}")
    
    # Return placeholder when not streaming
    try:
        import numpy as np
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if not streaming:
            text = "Click 'Start Video Stream'"
            cv2.putText(placeholder, text, (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            text = "Initializing video..."
            cv2.putText(placeholder, text, (180, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', placeholder)
        
        return Response(
            (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        print(f"❌ Placeholder error: {e}")
        return "Video feed error", 500

if __name__ == '__main__':
    print("🚀 Starting minimal video streaming test...")
    print(f"📹 Video source: {video_source}")
    
    # Test video file exists
    import os
    if os.path.exists(video_source):
        print("✅ Test video file found")
    else:
        print("❌ Test video file missing")
    
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)