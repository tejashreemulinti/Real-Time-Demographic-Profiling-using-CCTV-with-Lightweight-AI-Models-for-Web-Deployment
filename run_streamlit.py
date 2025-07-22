import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import threading
import time
import logging

from backend.video_processor import VideoProcessor
from backend.anonymizer import PrivacyModeManager
from backend.statistics import DemographicAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Real-Time Demographic Profiling",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #28a745;
    }
    
    .status-inactive {
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None
if 'privacy_manager' not in st.session_state:
    st.session_state.privacy_manager = PrivacyModeManager()
if 'analytics' not in st.session_state:
    st.session_state.analytics = DemographicAnalytics()
if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None
if 'statistics' not in st.session_state:
    st.session_state.statistics = {}


class StreamlitVideoHandler:
    """Handle video processing for Streamlit interface."""
    
    def __init__(self):
        self.latest_frame = None
        self.latest_statistics = {}
        
    def frame_callback(self, result):
        """Handle frame updates."""
        # Apply privacy mode
        processed_frame = result['frame']
        faces = result['faces']
        
        if st.session_state.privacy_manager.get_current_mode() != 'none':
            processed_frame = st.session_state.privacy_manager.apply_privacy_mode(
                processed_frame, faces
            )
        
        self.latest_frame = processed_frame
        
        # Add detections to analytics
        for face in faces:
            st.session_state.analytics.add_detection(face)
    
    def statistics_callback(self, statistics):
        """Handle statistics updates."""
        self.latest_statistics = statistics


# Global video handler
if 'video_handler' not in st.session_state:
    st.session_state.video_handler = StreamlitVideoHandler()


def initialize_system():
    """Initialize the video processing system."""
    try:
        if st.session_state.video_processor is None:
            st.session_state.video_processor = VideoProcessor(
                source=0,
                use_lightweight_models=True,
                max_faces=4
            )
            
            st.session_state.video_processor.set_callbacks(
                frame_callback=st.session_state.video_handler.frame_callback,
                statistics_callback=st.session_state.video_handler.statistics_callback
            )
            
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return False


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎥 Real-Time Demographic Profiling System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # System initialization
    if st.sidebar.button("🔧 Initialize System"):
        with st.spinner("Initializing system..."):
            if initialize_system():
                st.success("System initialized successfully!")
            else:
                st.error("Failed to initialize system!")
    
    # Video processing controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.processing_active):
            if st.session_state.video_processor is None:
                st.error("Please initialize the system first!")
            else:
                success = st.session_state.video_processor.start_processing(threaded=True)
                if success:
                    st.session_state.processing_active = True
                    st.success("Processing started!")
                else:
                    st.error("Failed to start processing!")
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.processing_active):
            if st.session_state.video_processor:
                st.session_state.video_processor.stop_processing()
                st.session_state.processing_active = False
                st.success("Processing stopped!")
    
    # Privacy mode selector
    st.sidebar.markdown("### 🔒 Privacy Settings")
    privacy_modes = st.session_state.privacy_manager.get_available_modes()
    current_mode = st.session_state.privacy_manager.get_current_mode()
    
    selected_mode = st.sidebar.selectbox(
        "Privacy Mode",
        privacy_modes,
        index=privacy_modes.index(current_mode)
    )
    
    if selected_mode != current_mode:
        st.session_state.privacy_manager.set_privacy_mode(selected_mode)
        st.sidebar.success(f"Privacy mode set to: {selected_mode}")
    
    # Analytics controls
    st.sidebar.markdown("### 📊 Analytics")
    
    if st.sidebar.button("📈 Export Data (CSV)"):
        if st.session_state.analytics.session_data:
            filename = st.session_state.analytics.export_to_csv()
            st.sidebar.success(f"Data exported to: {filename}")
        else:
            st.sidebar.warning("No data to export!")
    
    if st.sidebar.button("📋 Export Analysis (JSON)"):
        if st.session_state.analytics.session_data:
            filename = st.session_state.analytics.export_to_json()
            st.sidebar.success(f"Analysis exported to: {filename}")
        else:
            st.sidebar.warning("No data to export!")
    
    if st.sidebar.button("🗑️ Reset Statistics"):
        st.session_state.analytics.clear_data()
        st.sidebar.success("Statistics reset!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Video Feed")
        
        # Status indicator
        status_color = "active" if st.session_state.processing_active else "inactive"
        status_text = "Processing Active" if st.session_state.processing_active else "Processing Inactive"
        
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span class="status-indicator status-{status_color}"></span>
            <strong>{status_text}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Video frame placeholder
        frame_placeholder = st.empty()
        
        # Performance metrics
        if st.session_state.processing_active and st.session_state.video_processor:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                fps = st.session_state.video_processor.current_fps
                st.metric("FPS", f"{fps:.1f}")
            
            with metrics_col2:
                # This would need to be tracked separately
                st.metric("Faces", "0")
            
            with metrics_col3:
                avg_time = np.mean(st.session_state.video_processor.statistics['processing_times']) if st.session_state.video_processor.statistics['processing_times'] else 0
                st.metric("Avg Processing", f"{avg_time*1000:.1f}ms")
            
            with metrics_col4:
                runtime = time.time() - st.session_state.video_processor.statistics['session_start_time']
                st.metric("Runtime", f"{runtime:.0f}s")
    
    with col2:
        st.markdown("### 📊 Live Statistics")
        
        # Get current statistics
        if st.session_state.analytics.session_data:
            summary = st.session_state.analytics.get_demographic_summary()
            
            # Key metrics
            st.metric("Total Detections", summary.get('total_detections', 0))
            
            # Gender distribution
            if 'gender_analysis' in summary:
                gender_data = summary['gender_analysis']['distribution']
                if gender_data:
                    fig_gender = px.pie(
                        values=list(gender_data.values()),
                        names=list(gender_data.keys()),
                        title="Gender Distribution"
                    )
                    fig_gender.update_layout(height=300)
                    st.plotly_chart(fig_gender, use_container_width=True)
            
            # Age distribution
            if 'age_analysis' in summary:
                age_data = summary['age_analysis']['distribution']
                if age_data:
                    fig_age = px.bar(
                        x=list(age_data.keys()),
                        y=list(age_data.values()),
                        title="Age Distribution"
                    )
                    fig_age.update_layout(height=300)
                    st.plotly_chart(fig_age, use_container_width=True)
        
        else:
            st.info("No data available yet. Start processing to see statistics.")
    
    # Detailed Analytics Tab
    st.markdown("### 📈 Detailed Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "⏰ Temporal Analysis", "🎯 Confidence Analysis"])
    
    with tab1:
        if st.session_state.analytics.session_data:
            summary = st.session_state.analytics.get_demographic_summary()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Gender Analysis")
                if 'gender_analysis' in summary:
                    gender_stats = summary['gender_analysis']
                    st.write(f"**Most Common:** {gender_stats.get('most_common', 'Unknown')}")
                    st.write(f"**Diversity Index:** {gender_stats.get('diversity_index', 0):.3f}")
            
            with col2:
                st.markdown("#### Age Analysis")
                if 'age_analysis' in summary:
                    age_stats = summary['age_analysis']
                    st.write(f"**Most Common:** {age_stats.get('most_common', 'Unknown')}")
                    st.write(f"**Average Age:** {age_stats.get('average_age', 0):.1f}")
                    st.write(f"**Diversity Index:** {age_stats.get('diversity_index', 0):.3f}")
            
            with col3:
                st.markdown("#### Session Info")
                st.write(f"**Total Detections:** {summary.get('total_detections', 0)}")
                st.write(f"**Unique Sessions:** {summary.get('unique_sessions', 0)}")
                
                if 'date_range' in summary:
                    date_range = summary['date_range']
                    st.write(f"**Start:** {date_range.get('start', 'Unknown')}")
        
        else:
            st.info("Start processing to see detailed analytics.")
    
    with tab2:
        if st.session_state.analytics.session_data:
            summary = st.session_state.analytics.get_demographic_summary()
            
            if 'temporal_analysis' in summary:
                temporal = summary['temporal_analysis']
                
                # Hourly distribution
                if 'hourly_distribution' in temporal:
                    hourly_data = temporal['hourly_distribution']
                    if hourly_data:
                        fig_hourly = px.line(
                            x=list(hourly_data.keys()),
                            y=list(hourly_data.values()),
                            title="Hourly Activity Distribution",
                            labels={'x': 'Hour of Day', 'y': 'Detection Count'}
                        )
                        st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Peak hours info
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Peak Hour:** {temporal.get('peak_hour', 'Unknown')}")
                    st.write(f"**Peak Day:** {temporal.get('peak_day', 'Unknown')}")
                
                with col2:
                    if 'busiest_hours' in temporal:
                        st.write("**Busiest Hours:**")
                        for hour, count in temporal['busiest_hours'].items():
                            st.write(f"  {hour}:00 - {count} detections")
        
        else:
            st.info("Start processing to see temporal analysis.")
    
    with tab3:
        if st.session_state.analytics.session_data:
            summary = st.session_state.analytics.get_demographic_summary()
            
            if 'confidence_analysis' in summary:
                confidence = summary['confidence_analysis']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Age Confidence")
                    age_conf = confidence.get('age_confidence', {})
                    st.write(f"**Mean:** {age_conf.get('mean', 0):.3f}")
                    st.write(f"**Std:** {age_conf.get('std', 0):.3f}")
                    st.write(f"**Min:** {age_conf.get('min', 0):.3f}")
                    st.write(f"**Max:** {age_conf.get('max', 0):.3f}")
                
                with col2:
                    st.markdown("#### Gender Confidence")
                    gender_conf = confidence.get('gender_confidence', {})
                    st.write(f"**Mean:** {gender_conf.get('mean', 0):.3f}")
                    st.write(f"**Std:** {gender_conf.get('std', 0):.3f}")
                    st.write(f"**Min:** {gender_conf.get('min', 0):.3f}")
                    st.write(f"**Max:** {gender_conf.get('max', 0):.3f}")
                
                # Overall confidence
                overall = confidence.get('overall_confidence', {})
                st.markdown("#### Overall Performance")
                st.write(f"**Overall Mean Confidence:** {overall.get('mean', 0):.3f}")
                st.write(f"**High Confidence Rate:** {overall.get('high_confidence_rate', 0):.1f}%")
        
        else:
            st.info("Start processing to see confidence analysis.")
    
    # Real-time frame updates
    if st.session_state.processing_active and st.session_state.video_handler.latest_frame is not None:
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(st.session_state.video_handler.latest_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    # Auto-refresh every 2 seconds when processing
    if st.session_state.processing_active:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()