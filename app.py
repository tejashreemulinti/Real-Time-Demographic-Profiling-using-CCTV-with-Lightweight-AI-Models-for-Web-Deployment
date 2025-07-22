from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
import threading
import time
import json
import logging
from typing import Dict, Any

from backend.video_processor import VideoProcessor
from backend.anonymizer import PrivacyModeManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'demographic_profiling_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
video_processor = None
privacy_manager = PrivacyModeManager()
processing_active = False
current_statistics = {}
stream_clients = set()


class VideoStreamManager:
    """Manages video streaming to multiple clients."""
    
    def __init__(self):
        self.active_streams = 0
        self.frame_buffer = None
        self.statistics_buffer = None
        
    def add_client(self):
        self.active_streams += 1
        logger.info(f"Client connected. Active streams: {self.active_streams}")
        
    def remove_client(self):
        self.active_streams = max(0, self.active_streams - 1)
        logger.info(f"Client disconnected. Active streams: {self.active_streams}")
        
    def update_frame(self, frame, statistics):
        self.frame_buffer = frame
        self.statistics_buffer = statistics


stream_manager = VideoStreamManager()


def initialize_video_processor():
    """Initialize the video processor."""
    global video_processor
    
    try:
        video_processor = VideoProcessor(
            source=0,  # Use webcam by default
            use_lightweight_models=True,
            max_faces=4
        )
        
        # Set callbacks for real-time updates
        video_processor.set_callbacks(
            frame_callback=frame_update_callback,
            statistics_callback=statistics_update_callback
        )
        
        logger.info("Video processor initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize video processor: {e}")
        return False


def frame_update_callback(result: Dict[Any, Any]):
    """Callback for frame updates."""
    global current_statistics
    
    # Apply privacy mode if enabled
    processed_frame = result['frame']
    faces = result['faces']
    
    if privacy_manager.get_current_mode() != 'none':
        processed_frame = privacy_manager.apply_privacy_mode(processed_frame, faces)
    
    # Update stream manager
    stream_manager.update_frame(processed_frame, result['statistics'])
    
    # Emit to connected clients via WebSocket
    if stream_manager.active_streams > 0:
        frame_data = encode_frame_to_base64(processed_frame)
        socketio.emit('frame_update', {
            'frame': frame_data,
            'faces': len(faces),
            'fps': result['fps'],
            'processing_time': result['processing_time']
        })


def statistics_update_callback(statistics: Dict[Any, Any]):
    """Callback for statistics updates."""
    global current_statistics
    current_statistics = statistics
    
    # Emit statistics to connected clients
    socketio.emit('statistics_update', statistics)


def encode_frame_to_base64(frame):
    """Encode frame to base64 for web transmission."""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return None


def generate_video_stream():
    """Generator function for video streaming."""
    while processing_active and video_processor:
        frame = video_processor.get_current_frame()
        
        if frame is not None:
            # Apply privacy mode
            if privacy_manager.get_current_mode() != 'none':
                faces = []  # Get faces from processor if needed
                frame = privacy_manager.apply_privacy_mode(frame, faces)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


# Flask Routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Analytics dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/start_processing', methods=['POST'])
def start_processing():
    """Start video processing."""
    global processing_active, video_processor
    
    try:
        if not video_processor:
            if not initialize_video_processor():
                return jsonify({'error': 'Failed to initialize video processor'}), 500
        
        if not processing_active:
            success = video_processor.start_processing(threaded=True)
            if success:
                processing_active = True
                return jsonify({'status': 'success', 'message': 'Processing started'})
            else:
                return jsonify({'error': 'Failed to start processing'}), 500
        else:
            return jsonify({'status': 'already_running', 'message': 'Processing already active'})
            
    except Exception as e:
        logger.error(f"Error starting processing: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    """Stop video processing."""
    global processing_active, video_processor
    
    try:
        processing_active = False
        
        if video_processor:
            video_processor.stop_processing()
        
        return jsonify({'status': 'success', 'message': 'Processing stopped'})
        
    except Exception as e:
        logger.error(f"Error stopping processing: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def get_status():
    """Get current system status."""
    return jsonify({
        'processing_active': processing_active,
        'privacy_mode': privacy_manager.get_current_mode(),
        'active_streams': stream_manager.active_streams,
        'video_processor_ready': video_processor is not None
    })


@app.route('/api/statistics')
def get_statistics():
    """Get current statistics."""
    global current_statistics
    
    if video_processor:
        stats = video_processor.get_statistics()
        return jsonify(stats)
    else:
        return jsonify(current_statistics)


@app.route('/api/reset_statistics', methods=['POST'])
def reset_statistics():
    """Reset statistics."""
    try:
        if video_processor:
            video_processor.reset_statistics()
        
        global current_statistics
        current_statistics = {}
        
        return jsonify({'status': 'success', 'message': 'Statistics reset'})
        
    except Exception as e:
        logger.error(f"Error resetting statistics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/privacy_mode', methods=['GET', 'POST'])
def privacy_mode():
    """Get or set privacy mode."""
    if request.method == 'GET':
        return jsonify({
            'current_mode': privacy_manager.get_current_mode(),
            'available_modes': privacy_manager.get_available_modes()
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            mode = data.get('mode')
            
            if mode:
                privacy_manager.set_privacy_mode(mode)
                return jsonify({'status': 'success', 'mode': mode})
            else:
                return jsonify({'error': 'Mode not specified'}), 400
                
        except Exception as e:
            logger.error(f"Error setting privacy mode: {e}")
            return jsonify({'error': str(e)}), 500


@app.route('/api/screenshot', methods=['POST'])
def capture_screenshot():
    """Capture a screenshot of current frame."""
    try:
        if video_processor and processing_active:
            filename = video_processor.capture_screenshot()
            return jsonify({'status': 'success', 'filename': filename})
        else:
            return jsonify({'error': 'Video processing not active'}), 400
            
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    if processing_active:
        return Response(
            generate_video_stream(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    else:
        # Return placeholder image when not processing
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Video Processing Stopped', (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', placeholder)
        return Response(
            (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )


# WebSocket Events
@socketio.on('connect')
def on_connect():
    """Handle client connection."""
    stream_manager.add_client()
    
    # Send current status to new client
    emit('status_update', {
        'processing_active': processing_active,
        'privacy_mode': privacy_manager.get_current_mode()
    })
    
    # Send current statistics if available
    if current_statistics:
        emit('statistics_update', current_statistics)


@socketio.on('disconnect')
def on_disconnect():
    """Handle client disconnection."""
    stream_manager.remove_client()


@socketio.on('request_frame')
def on_request_frame():
    """Handle frame request from client."""
    if stream_manager.frame_buffer is not None:
        frame_data = encode_frame_to_base64(stream_manager.frame_buffer)
        if frame_data:
            emit('frame_update', {
                'frame': frame_data,
                'statistics': stream_manager.statistics_buffer
            })


@socketio.on('set_privacy_mode')
def on_set_privacy_mode(data):
    """Handle privacy mode change via WebSocket."""
    try:
        mode = data.get('mode')
        if mode:
            privacy_manager.set_privacy_mode(mode)
            emit('privacy_mode_changed', {'mode': mode}, broadcast=True)
    except Exception as e:
        logger.error(f"Error setting privacy mode via WebSocket: {e}")
        emit('error', {'message': str(e)})


# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


# Cleanup function
def cleanup():
    """Cleanup resources on shutdown."""
    global video_processor, processing_active
    
    processing_active = False
    
    if video_processor:
        video_processor.stop_processing()


if __name__ == '__main__':
    try:
        # Initialize video processor
        logger.info("Starting Real-Time Demographic Profiling System...")
        
        # Run the Flask app with SocketIO
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        cleanup()
    except Exception as e:
        logger.error(f"Application error: {e}")
        cleanup()