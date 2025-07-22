# Real-Time Demographic Profiling System 🎥

A lightweight, real-time AI system for demographic analysis from CCTV/webcam feeds using MediaPipe and lightweight ML models. Features face detection, age/gender estimation, privacy controls, and web-based analytics dashboard.

## 🌟 Features

### Core Functionality
- **Real-time Face Detection**: Using MediaPipe for CPU-optimized performance
- **Age & Gender Estimation**: Lightweight MobileNetV2-based models
- **Multi-face Processing**: Handle up to 4 faces simultaneously
- **Privacy Protection**: Multiple anonymization modes (blur, pixelate, mask, etc.)
- **Live Analytics**: Real-time demographic statistics and insights

### Web Interface
- **Flask-based Dashboard**: Real-time video streaming with WebSocket support
- **Streamlit Alternative**: Simple interface for quick deployment
- **Interactive Charts**: Live demographic distribution visualizations
- **Export Capabilities**: CSV/JSON data export for further analysis

### Performance Optimized
- **CPU-only Operation**: No GPU required, runs on modest hardware
- **Target Performance**: ≥10 FPS on CPU for up to 4 faces
- **Lightweight Models**: Optimized for edge deployment
- **Threading Support**: Non-blocking video processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam or video source
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd real-time-demographic-profiling
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optimize for performance** (recommended):
   ```bash
   python optimize_performance.py
   ```

4. **Run the Flask application**
   ```bash
   python app.py
   ```
   
   Or run the Streamlit interface:
   ```bash
   streamlit run run_streamlit.py
   ```

5. **Open your browser**
   - Flask: http://localhost:5000
   - Streamlit: http://localhost:8501

## 🎯 Usage

### Flask Web Interface

1. **Initialize System**: Click "Start Processing" to begin video analysis
2. **View Live Feed**: Real-time video stream with face detection annotations
3. **Monitor Statistics**: Live demographic charts and performance metrics
4. **Privacy Controls**: Select anonymization mode from dropdown
5. **Export Data**: Download CSV/JSON reports for analysis

### Streamlit Interface

1. **Initialize System**: Click "Initialize System" in sidebar
2. **Start Processing**: Use sidebar controls to start/stop analysis
3. **Configure Privacy**: Select privacy mode from sidebar
4. **View Analytics**: Multiple tabs with detailed demographic insights
5. **Export Results**: Use sidebar buttons to export data

### Privacy Modes

- **None**: No privacy protection
- **Low**: Light blurring (30% intensity)
- **Medium**: Selective protection (protects minors)
- **High**: Strong masking (80% intensity)
- **Maximum**: Black bar censorship (100% intensity)

## 📊 Analytics Features

### Real-time Metrics
- Face detection count
- Processing FPS
- Gender distribution
- Age group distribution
- Confidence scores

### Advanced Analytics
- Temporal pattern analysis
- Peak activity hours
- Demographic diversity indices
- Confidence distribution analysis
- Data export capabilities

### Visualization Types
- Pie charts (gender distribution)
- Bar charts (age groups)
- Line graphs (temporal patterns)
- Heatmaps (activity patterns)
- Histograms (confidence scores)

## 🔧 Configuration

### Video Source Options
```python
# Webcam (default)
source = 0

# USB camera
source = 1

# IP camera
source = "http://192.168.1.100:8080/video"

# Video file
source = "path/to/video.mp4"
```

### Model Configuration
```python
# Use lightweight models (recommended)
use_lightweight_models = True

# Maximum faces to detect
max_faces = 4

# Detection confidence threshold
confidence_threshold = 0.7
```

### Performance Tuning
```python
# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
```

## 🏗️ Architecture

### Project Structure
```
real-time-demographic-profiling/
├── app.py                          # Main Flask application
├── run_streamlit.py               # Streamlit interface
├── requirements.txt               # Dependencies
├── backend/
│   ├── face_detector.py           # MediaPipe face detection
│   ├── age_gender_estimator.py    # ML models for demographics
│   ├── video_processor.py         # Main processing pipeline
│   ├── anonymizer.py              # Privacy protection
│   └── statistics.py              # Analytics engine
├── templates/
│   └── index.html                 # Web interface
└── models/                        # ML model storage
```

### Core Components

1. **Face Detector**: MediaPipe-based face detection
2. **Demographic Estimator**: MobileNetV2 age/gender models
3. **Video Processor**: Main processing pipeline with threading
4. **Anonymizer**: Privacy protection with multiple modes
5. **Analytics Engine**: Statistical analysis and export
6. **Web Interface**: Flask + WebSocket real-time dashboard

## 🔬 Technical Details

### AI Models
- **Face Detection**: MediaPipe Face Detection (BlazeFace)
- **Age Estimation**: MobileNetV2 + custom classifier (8 age groups)
- **Gender Estimation**: MobileNetV2 + binary classifier
- **Input Size**: 224x224 pixels for demographic models

### Performance Benchmarks
- **CPU Only**: 10-15 FPS on modern laptops
- **Memory Usage**: ~500MB RAM
- **Processing Time**: 30-100ms per frame
- **Model Size**: <50MB total

### Age Groups
- 0-10 years
- 11-20 years
- 21-30 years
- 31-40 years
- 41-50 years
- 51-60 years
- 61-70 years
- 71+ years

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Docker Deployment
```bash
docker build -t demographic-profiling .
docker run -p 5000:5000 --device /dev/video0 demographic-profiling
```

### Cloud Deployment
- Compatible with AWS, GCP, Azure
- Requires camera access or video input
- Use lightweight compute instances

### Edge Deployment
- Raspberry Pi 4+ recommended
- Jetson Nano for enhanced performance
- USB webcam or CSI camera support

## 📈 Future Enhancements

### Planned Features
- [ ] Emotion recognition
- [ ] Behavioral analysis
- [ ] Crowd density estimation
- [ ] CCTV NVR integration
- [ ] Cloud-based processing
- [ ] Mobile app interface
- [ ] Advanced reporting
- [ ] Multi-camera support

### Model Improvements
- [ ] Better age estimation accuracy
- [ ] Ethnicity detection
- [ ] Pose estimation
- [ ] Facial landmarks
- [ ] ONNX model optimization

## 🛡️ Privacy & Ethics

### Privacy Features
- Built-in anonymization options
- Selective protection (age-based)
- No data storage by default
- Local processing only
- Configurable retention policies

### Ethical Considerations
- Designed for legitimate surveillance needs
- Includes privacy protection mechanisms
- Transparent about data collection
- User consent mechanisms
- Compliance with local regulations

### Security
- No cloud dependencies by default
- Local model execution
- Encrypted data transmission (HTTPS)
- Access control mechanisms
- Audit logging capabilities

## 🐛 Troubleshooting & Optimization

### Performance Issues

**JSON Serialization Error: "Object of type deque is not JSON serializable"**
```bash
# This has been fixed in the latest version
# Update your code or restart the application
python app.py
```

**Low FPS or High Processing Time**
```bash
# Run the optimization script first
python optimize_performance.py

# Then start the application
python app.py
```

**TensorFlow/Model Loading Issues**
```bash
# Set environment variables for better TensorFlow performance
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=1

# Or run the optimization script
python optimize_performance.py
```

### Speed Optimizations

1. **Run Performance Optimization**:
   ```bash
   python optimize_performance.py
   ```

2. **Adjust Quality Mode** in `config.py`:
   ```python
   QUALITY_MODE = "speed"  # For maximum speed
   QUALITY_MODE = "balanced"  # Default
   QUALITY_MODE = "quality"  # For best accuracy
   ```

3. **Camera Optimization**:
   - Use USB 3.0 camera
   - Reduce resolution to 640x480
   - Ensure good lighting
   - Keep faces within 2 meters

### Accuracy Improvements

1. **Use Improved Models**:
   - The system now uses EfficientNetB0-based models
   - More detailed age groups (10 instead of 8)
   - Better preprocessing with histogram equalization

2. **Optimize Detection Settings**:
   ```python
   # In config.py
   FACE_DETECTION_CONFIDENCE = 0.7  # Higher for better accuracy
   MAX_FACES = 6  # More faces for crowded scenes
   ```

3. **Enable Batch Processing**:
   - Automatic batch processing for multiple faces
   - Concurrent age/gender estimation
   - Model warm-up for faster inference

### Common Issues

**Camera not detected**
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"
```

**Memory issues**
```bash
# Reduce batch size in config.py
BATCH_SIZE = 2  # Instead of 4
MAX_FACES = 2   # Instead of 4
```

**Template not found errors**
```bash
# This has been fixed - error handlers now return JSON
# No additional action needed
```

**Model accuracy issues**
```bash
# Use quality mode for better accuracy
# Edit config.py and set:
QUALITY_MODE = "quality"
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

- Create an issue for bug reports
- Use discussions for feature requests
- Check the wiki for detailed documentation
- Join our community Discord server

## 🙏 Acknowledgments

- MediaPipe team for face detection
- TensorFlow team for ML frameworks
- OpenCV community for computer vision tools
- Flask and Streamlit for web frameworks

---

**⚠️ Important**: This system is designed for legitimate surveillance and analytics use cases. Please ensure compliance with local privacy laws and regulations when deploying in production environments.