#!/usr/bin/env python3
"""
Real-Time Demographic Analysis System
Captures video and detects age & gender in real-time
"""

from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import numpy as np
import time
import threading
import json
from collections import defaultdict, deque
from datetime import datetime
import random

app = Flask(__name__)

# Global variables
video_source = "test_video.avi"  # Change to 0 for real camera
streaming = False
processing = False
current_frame = None
cap = None
face_cascade = None
demographics_data = defaultdict(int)
session_stats = {
    'total_faces': 0,
    'unique_faces': set(),
    'gender_stats': {'Male': 0, 'Female': 0},
    'age_stats': defaultdict(int),
    'session_start': None,
    'fps_history': deque(maxlen=30)
}

# Age and gender estimation (simplified for demo)
AGE_GROUPS = ["0-5", "6-12", "13-17", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
GENDERS = ["Male", "Female"]

def init_video():
    """Initialize video capture and face detection."""
    global cap, face_cascade
    
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ Failed to open video source: {video_source}")
            return False
        
        # Initialize face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print(f"✅ Video initialized: {video_source}")
        return True
        
    except Exception as e:
        print(f"❌ Video init error: {e}")
        return False

def estimate_demographics(face_crop):
    """
    Estimate age and gender from face crop.
    This is a simplified simulation for demo purposes.
    In production, you'd use trained ML models.
    """
    # Simulate processing time
    time.sleep(0.01)
    
    # Simple heuristics based on face characteristics (for demo)
    height, width = face_crop.shape[:2]
    
    # Simulate age estimation based on face size and texture
    face_area = height * width
    
    if face_area < 3000:
        age_group = random.choice(["0-5", "6-12"])
    elif face_area < 6000:
        age_group = random.choice(["13-17", "18-25"])
    elif face_area < 10000:
        age_group = random.choice(["18-25", "26-35", "36-45"])
    else:
        age_group = random.choice(["36-45", "46-55", "56-65"])
    
    # Simulate gender estimation
    gender = random.choice(GENDERS)
    
    # Calculate average brightness (simplified feature)
    brightness = np.mean(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY))
    
    # Confidence scores (simulated)
    age_confidence = random.uniform(0.75, 0.95)
    gender_confidence = random.uniform(0.70, 0.90)
    
    return {
        'age_group': age_group,
        'age_confidence': age_confidence,
        'gender': gender,
        'gender_confidence': gender_confidence,
        'brightness': brightness
    }

def detect_faces_and_demographics(frame):
    """Detect faces and estimate demographics."""
    global face_cascade, session_stats
    
    if face_cascade is None:
        return frame, []
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_data = []
    
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face region
        face_crop = frame[y:y+h, x:x+w]
        
        if face_crop.size > 0:
            # Estimate demographics
            demographics = estimate_demographics(face_crop)
            
            # Store face data
            face_info = {
                'id': i + 1,
                'bbox': (x, y, w, h),
                'demographics': demographics,
                'center': (x + w//2, y + h//2)
            }
            face_data.append(face_info)
            
            # Update session statistics
            session_stats['total_faces'] += 1
            face_key = f"{x}_{y}_{w}_{h}"  # Simple face tracking
            if face_key not in session_stats['unique_faces']:
                session_stats['unique_faces'].add(face_key)
                session_stats['gender_stats'][demographics['gender']] += 1
                session_stats['age_stats'][demographics['age_group']] += 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw demographics info
            age_text = f"Age: {demographics['age_group']} ({demographics['age_confidence']:.1%})"
            gender_text = f"{demographics['gender']} ({demographics['gender_confidence']:.1%})"
            
            # Text background
            cv2.rectangle(frame, (x, y-60), (x + max(len(age_text), len(gender_text)) * 8, y), (0, 0, 0), -1)
            
            # Text overlay
            cv2.putText(frame, age_text, (x + 2, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, gender_text, (x + 2, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Face ID
            cv2.putText(frame, f"#{face_info['id']}", (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame, face_data

def generate_frames():
    """Generate video frames with demographic analysis."""
    global cap, streaming, processing, session_stats
    
    if not cap or not cap.isOpened():
        return
    
    frame_count = 0
    fps_start = time.time()
    
    while streaming:
        ret, frame = cap.read()
        
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            
        if ret:
            frame_count += 1
            original_frame = frame.copy()
            
            # Process demographics if enabled
            if processing:
                frame, face_data = detect_faces_and_demographics(frame)
                
                # Add processing info
                cv2.putText(frame, f'Faces Detected: {len(face_data)}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'Total Analyzed: {len(session_stats["unique_faces"])}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, 'Click "Start Analysis" to begin', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Calculate and display FPS
            if frame_count % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - fps_start)
                session_stats['fps_history'].append(fps)
                fps_start = current_time
            
            current_fps = session_stats['fps_history'][-1] if session_stats['fps_history'] else 0
            cv2.putText(frame, f'FPS: {current_fps:.1f}', (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (frame.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
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
    <title>Real-Time Demographic Analysis</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 1.2em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .video-section {
            background: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .video-container {
            position: relative;
            border: 3px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            background: #000;
        }
        
        #videoStream {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #95a5a6, #7f8c8d);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .stats-section {
            background: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .stats-header {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .stats-header h3 {
            color: #2c3e50;
            font-size: 1.5em;
            margin-bottom: 5px;
        }
        
        .status-indicator {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }
        
        .status-ready {
            background: #d4edda;
            color: #155724;
        }
        
        .status-streaming {
            background: #cce7ff;
            color: #004085;
        }
        
        .status-analyzing {
            background: #fff3cd;
            color: #856404;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .chart-container {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .chart-title {
            text-align: center;
            margin-bottom: 15px;
            color: #495057;
            font-weight: 600;
        }
        
        .chart-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .chart-label {
            width: 80px;
            font-size: 0.9em;
            color: #6c757d;
        }
        
        .chart-progress {
            flex: 1;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            margin: 0 10px;
            overflow: hidden;
        }
        
        .chart-fill {
            height: 100%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .chart-value {
            width: 40px;
            text-align: right;
            font-size: 0.9em;
            color: #495057;
            font-weight: 600;
        }
        
        .session-info {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: center;
        }
        
        .session-time {
            font-size: 1.2em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 200px;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Real-Time Demographic Analysis</h1>
            <p>Advanced AI-powered age and gender detection system</p>
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img id="videoStream" src="/video_feed" alt="Video Stream">
                </div>
                
                <div class="controls">
                    <button id="startStreamBtn" class="btn btn-primary" onclick="startStream()">
                        🎥 Start Camera
                    </button>
                    <button id="stopStreamBtn" class="btn btn-danger" onclick="stopStream()" disabled>
                        ⏹️ Stop Camera
                    </button>
                    <button id="startAnalysisBtn" class="btn btn-secondary" onclick="startAnalysis()" disabled>
                        🧠 Start Analysis
                    </button>
                    <button id="stopAnalysisBtn" class="btn btn-danger" onclick="stopAnalysis()" disabled>
                        🛑 Stop Analysis
                    </button>
                    <button class="btn btn-secondary" onclick="resetStats()">
                        🔄 Reset Stats
                    </button>
                </div>
            </div>
            
            <div class="stats-section">
                <div class="stats-header">
                    <h3>📊 Live Analytics</h3>
                    <div id="statusIndicator" class="status-indicator status-ready">
                        Ready to Start
                    </div>
                </div>
                
                <div class="metric-card">
                    <div id="totalFaces" class="metric-value">0</div>
                    <div class="metric-label">Total Faces Analyzed</div>
                </div>
                
                <div class="metric-card">
                    <div id="currentFPS" class="metric-value">0</div>
                    <div class="metric-label">Current FPS</div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">👥 Gender Distribution</div>
                    <div class="chart-bar">
                        <div class="chart-label">Male</div>
                        <div class="chart-progress">
                            <div id="maleBar" class="chart-fill" style="width: 0%"></div>
                        </div>
                        <div id="maleCount" class="chart-value">0</div>
                    </div>
                    <div class="chart-bar">
                        <div class="chart-label">Female</div>
                        <div class="chart-progress">
                            <div id="femaleBar" class="chart-fill" style="width: 0%"></div>
                        </div>
                        <div id="femaleCount" class="chart-value">0</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">🎂 Age Distribution</div>
                    <div id="ageChart"></div>
                </div>
                
                <div class="session-info">
                    <div class="session-time" id="sessionTime">00:00:00</div>
                    <div style="color: #6c757d; font-size: 0.9em;">Session Duration</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let streaming = false;
        let analyzing = false;
        let sessionStartTime = null;
        let statsInterval = null;
        let timeInterval = null;
        
        function updateStatus(message, type) {
            const indicator = document.getElementById('statusIndicator');
            indicator.textContent = message;
            indicator.className = 'status-indicator status-' + type;
        }
        
        function updateButtons() {
            document.getElementById('startStreamBtn').disabled = streaming;
            document.getElementById('stopStreamBtn').disabled = !streaming;
            document.getElementById('startAnalysisBtn').disabled = !streaming || analyzing;
            document.getElementById('stopAnalysisBtn').disabled = !analyzing;
        }
        
        function startStream() {
            fetch('/start_stream', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    streaming = true;
                    updateStatus('🎥 Camera Active', 'streaming');
                    updateButtons();
                    refreshVideo();
                } else {
                    alert('Failed to start camera: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error starting camera');
            });
        }
        
        function stopStream() {
            fetch('/stop_stream', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                streaming = false;
                analyzing = false;
                updateStatus('📷 Camera Stopped', 'ready');
                updateButtons();
                stopStatsUpdates();
            });
        }
        
        function startAnalysis() {
            fetch('/start_analysis', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    analyzing = true;
                    updateStatus('🧠 Analyzing Demographics', 'analyzing');
                    updateButtons();
                    startStatsUpdates();
                    startSessionTimer();
                } else {
                    alert('Failed to start analysis: ' + data.error);
                }
            });
        }
        
        function stopAnalysis() {
            fetch('/stop_analysis', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                analyzing = false;
                updateStatus('🎥 Camera Active', 'streaming');
                updateButtons();
                stopStatsUpdates();
            });
        }
        
        function resetStats() {
            fetch('/reset_stats', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                updateStatsDisplay({
                    total_faces: 0,
                    gender_stats: {Male: 0, Female: 0},
                    age_stats: {},
                    fps: 0
                });
                sessionStartTime = Date.now();
            });
        }
        
        function refreshVideo() {
            const video = document.getElementById('videoStream');
            video.src = '/video_feed?' + new Date().getTime();
        }
        
        function startStatsUpdates() {
            sessionStartTime = Date.now();
            statsInterval = setInterval(updateStats, 1000);
        }
        
        function stopStatsUpdates() {
            if (statsInterval) {
                clearInterval(statsInterval);
                statsInterval = null;
            }
        }
        
        function startSessionTimer() {
            timeInterval = setInterval(updateSessionTime, 1000);
        }
        
        function updateSessionTime() {
            if (sessionStartTime) {
                const elapsed = Date.now() - sessionStartTime;
                const hours = Math.floor(elapsed / 3600000);
                const minutes = Math.floor((elapsed % 3600000) / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                
                const timeStr = 
                    hours.toString().padStart(2, '0') + ':' +
                    minutes.toString().padStart(2, '0') + ':' +
                    seconds.toString().padStart(2, '0');
                
                document.getElementById('sessionTime').textContent = timeStr;
            }
        }
        
        function updateStats() {
            fetch('/get_stats')
            .then(response => response.json())
            .then(data => {
                updateStatsDisplay(data);
            })
            .catch(error => {
                console.error('Error fetching stats:', error);
            });
        }
        
        function updateStatsDisplay(stats) {
            // Update total faces
            document.getElementById('totalFaces').textContent = stats.total_faces || 0;
            
            // Update FPS
            document.getElementById('currentFPS').textContent = (stats.fps || 0).toFixed(1);
            
            // Update gender distribution
            const maleCount = stats.gender_stats?.Male || 0;
            const femaleCount = stats.gender_stats?.Female || 0;
            const totalGender = maleCount + femaleCount;
            
            document.getElementById('maleCount').textContent = maleCount;
            document.getElementById('femaleCount').textContent = femaleCount;
            
            if (totalGender > 0) {
                document.getElementById('maleBar').style.width = (maleCount / totalGender * 100) + '%';
                document.getElementById('femaleBar').style.width = (femaleCount / totalGender * 100) + '%';
            }
            
            // Update age distribution
            updateAgeChart(stats.age_stats || {});
        }
        
        function updateAgeChart(ageStats) {
            const ageChart = document.getElementById('ageChart');
            const ageGroups = ["0-5", "6-12", "13-17", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"];
            
            const totalAge = Object.values(ageStats).reduce((sum, count) => sum + count, 0);
            
            ageChart.innerHTML = '';
            
            ageGroups.forEach(group => {
                const count = ageStats[group] || 0;
                const percentage = totalAge > 0 ? (count / totalAge * 100) : 0;
                
                const barDiv = document.createElement('div');
                barDiv.className = 'chart-bar';
                barDiv.innerHTML = `
                    <div class="chart-label">${group}</div>
                    <div class="chart-progress">
                        <div class="chart-fill" style="width: ${percentage}%"></div>
                    </div>
                    <div class="chart-value">${count}</div>
                `;
                ageChart.appendChild(barDiv);
            });
        }
        
        // Initialize
        updateButtons();
        
        // Auto-refresh video every 30 seconds
        setInterval(() => {
            if (streaming) {
                refreshVideo();
            }
        }, 30000);
        
        // Start session timer immediately
        startSessionTimer();
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
            return jsonify({'success': False, 'error': 'Failed to initialize video'})
        
        streaming = True
        session_stats['session_start'] = datetime.now()
        print("✅ Video streaming started")
        
        return jsonify({'success': True, 'message': 'Streaming started'})
        
    except Exception as e:
        print(f"❌ Start stream error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop video streaming."""
    global streaming, processing, cap
    
    streaming = False
    processing = False
    
    if cap:
        cap.release()
        cap = None
    
    print("🔴 Video streaming stopped")
    return jsonify({'success': True, 'message': 'Streaming stopped'})

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Start demographic analysis."""
    global processing
    
    if not streaming:
        return jsonify({'success': False, 'error': 'Video stream not active'})
    
    processing = True
    session_stats['session_start'] = datetime.now()
    print("🧠 Demographic analysis started")
    
    return jsonify({'success': True, 'message': 'Analysis started'})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    """Stop demographic analysis."""
    global processing
    
    processing = False
    print("🛑 Demographic analysis stopped")
    
    return jsonify({'success': True, 'message': 'Analysis stopped'})

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset statistics."""
    global session_stats
    
    session_stats = {
        'total_faces': 0,
        'unique_faces': set(),
        'gender_stats': {'Male': 0, 'Female': 0},
        'age_stats': defaultdict(int),
        'session_start': datetime.now(),
        'fps_history': deque(maxlen=30)
    }
    
    print("📊 Statistics reset")
    return jsonify({'success': True, 'message': 'Statistics reset'})

@app.route('/get_stats')
def get_stats():
    """Get current statistics."""
    current_fps = session_stats['fps_history'][-1] if session_stats['fps_history'] else 0
    
    return jsonify({
        'total_faces': len(session_stats['unique_faces']),
        'gender_stats': dict(session_stats['gender_stats']),
        'age_stats': dict(session_stats['age_stats']),
        'fps': current_fps,
        'session_duration': (datetime.now() - session_stats['session_start']).total_seconds() if session_stats['session_start'] else 0
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    global streaming
    
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
    
    # Return placeholder
    try:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Click 'Start Camera' to begin", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', placeholder)
        return Response(
            (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        return "Video feed error", 500

if __name__ == '__main__':
    print("🚀 Starting Real-Time Demographic Analysis System...")
    print(f"📹 Video source: {video_source}")
    print("🌐 Open browser: http://localhost:5003")
    print("\n📋 Instructions:")
    print("1. Click 'Start Camera' to begin video streaming")
    print("2. Click 'Start Analysis' to enable demographic detection")
    print("3. View real-time statistics in the sidebar")
    
    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True)