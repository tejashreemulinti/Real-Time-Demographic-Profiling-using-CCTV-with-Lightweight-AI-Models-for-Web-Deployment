#!/usr/bin/env python3
"""
Minimal test Flask app to debug video processing issue
"""

from flask import Flask, Response, jsonify, render_template_string
import cv2
import numpy as np
import time
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
processing_active = False
camera = None
current_frame = None

def init_camera():
    """Initialize camera."""
    global camera
    try:
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            logger.info("Camera initialized successfully")
            return True
        else:
            logger.error("Failed to open camera")
            return False
    except Exception as e:
        logger.error(f"Camera init error: {e}")
        return False

def generate_frames():
    """Generate video frames."""
    global camera, current_frame
    
    while processing_active and camera and camera.isOpened():
        ret, frame = camera.read()
        
        if ret:
            current_frame = frame
            
            # Add text overlay
            cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Minimal Video Test</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .video-container { text-align: center; margin: 20px 0; }
            .controls { text-align: center; margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            .status { margin: 20px 0; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Minimal Video Processing Test</h1>
            
            <div class="status">
                <p><strong>Status:</strong> <span id="status">Not started</span></p>
                <p><strong>Camera:</strong> <span id="camera-status">Unknown</span></p>
            </div>
            
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="Video Feed" style="max-width: 100%; border: 2px solid #ddd;">
            </div>
            
            <div class="controls">
                <button onclick="startProcessing()">🟢 Start Processing</button>
                <button onclick="stopProcessing()">🔴 Stop Processing</button>
                <button onclick="checkStatus()">🔍 Check Status</button>
            </div>
        </div>
        
        <script>
            function startProcessing() {
                fetch('/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log('Start response:', data);
                    document.getElementById('status').textContent = data.message || 'Started';
                    // Force reload video feed
                    document.getElementById('videoFeed').src = '/video_feed?' + Date.now();
                })
                .catch(error => {
                    console.error('Start error:', error);
                    document.getElementById('status').textContent = 'Error starting';
                });
            }
            
            function stopProcessing() {
                fetch('/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log('Stop response:', data);
                    document.getElementById('status').textContent = data.message || 'Stopped';
                })
                .catch(error => {
                    console.error('Stop error:', error);
                });
            }
            
            function checkStatus() {
                fetch('/status')
                .then(response => response.json())
                .then(data => {
                    console.log('Status:', data);
                    document.getElementById('status').textContent = data.processing ? 'Running' : 'Stopped';
                    document.getElementById('camera-status').textContent = data.camera ? 'Ready' : 'Not ready';
                })
                .catch(error => {
                    console.error('Status error:', error);
                });
            }
            
            // Check status on page load
            window.onload = checkStatus;
        </script>
    </body>
    </html>
    ''')

@app.route('/start', methods=['POST'])
def start():
    global processing_active
    
    try:
        if not camera:
            if not init_camera():
                return jsonify({'error': 'Camera initialization failed'}), 500
        
        processing_active = True
        logger.info("Processing started")
        
        return jsonify({'success': True, 'message': 'Processing started'})
        
    except Exception as e:
        logger.error(f"Start error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stop', methods=['POST'])  
def stop():
    global processing_active
    
    processing_active = False
    logger.info("Processing stopped")
    
    return jsonify({'success': True, 'message': 'Processing stopped'})

@app.route('/status')
def status():
    return jsonify({
        'processing': processing_active,
        'camera': camera is not None and camera.isOpened() if camera else False
    })

@app.route('/video_feed')
def video_feed():
    global processing_active
    
    if processing_active:
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    else:
        # Return a placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Video Stopped', (200, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', placeholder)
        return Response(
            (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

if __name__ == '__main__':
    logger.info("Starting minimal video test server...")
    
    app.run(host='0.0.0.0', port=5001, debug=True)