import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemographicAnalytics:
    """
    Advanced analytics for demographic data with export capabilities.
    """
    
    def __init__(self):
        """Initialize the analytics module."""
        self.session_data = []
        self.hourly_stats = defaultdict(lambda: {
            'face_count': 0,
            'gender_distribution': Counter(),
            'age_distribution': Counter(),
            'avg_confidence': {'age': 0, 'gender': 0}
        })
        
    def add_detection(self, face_data: Dict, timestamp: Optional[datetime] = None):
        """
        Add a new face detection to the analytics.
        
        Args:
            face_data: Face detection data with demographics
            timestamp: Detection timestamp (current time if None)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        demographics = face_data.get('demographics', {})
        
        detection_record = {
            'timestamp': timestamp,
            'face_id': face_data.get('id', 0),
            'bbox': face_data.get('bbox', (0, 0, 0, 0)),
            'confidence': face_data.get('confidence', 0),
            'age_group': demographics.get('age_group', 'Unknown'),
            'age_confidence': demographics.get('age_confidence', 0),
            'gender': demographics.get('gender', 'Unknown'),
            'gender_confidence': demographics.get('gender_confidence', 0),
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'date': timestamp.date()
        }
        
        self.session_data.append(detection_record)
        
        # Update hourly statistics
        hour_key = timestamp.hour
        self.hourly_stats[hour_key]['face_count'] += 1
        self.hourly_stats[hour_key]['gender_distribution'][demographics.get('gender', 'Unknown')] += 1
        self.hourly_stats[hour_key]['age_distribution'][demographics.get('age_group', 'Unknown')] += 1
        
    def get_demographic_summary(self) -> Dict:
        """
        Get comprehensive demographic summary.
        
        Returns:
            Dictionary containing demographic analysis
        """
        if not self.session_data:
            return {'error': 'No data available'}
        
        df = pd.DataFrame(self.session_data)
        
        summary = {
            'total_detections': len(df),
            'unique_sessions': len(df['date'].unique()),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'gender_analysis': self._analyze_gender(df),
            'age_analysis': self._analyze_age(df),
            'temporal_analysis': self._analyze_temporal_patterns(df),
            'confidence_analysis': self._analyze_confidence(df),
            'peak_hours': self._get_peak_hours(df)
        }
        
        return summary
    
    def _analyze_gender(self, df: pd.DataFrame) -> Dict:
        """Analyze gender distribution."""
        gender_counts = df['gender'].value_counts()
        total = len(df)
        
        return {
            'distribution': gender_counts.to_dict(),
            'percentages': (gender_counts / total * 100).round(2).to_dict(),
            'most_common': gender_counts.index[0] if len(gender_counts) > 0 else 'Unknown',
            'diversity_index': self._calculate_diversity_index(gender_counts)
        }
    
    def _analyze_age(self, df: pd.DataFrame) -> Dict:
        """Analyze age distribution."""
        age_counts = df['age_group'].value_counts()
        total = len(df)
        
        # Convert age groups to numeric for additional analysis
        age_numeric = df['age_group'].map({
            '0-10': 5, '11-20': 15, '21-30': 25, '31-40': 35,
            '41-50': 45, '51-60': 55, '61-70': 65, '71+': 75
        }).dropna()
        
        return {
            'distribution': age_counts.to_dict(),
            'percentages': (age_counts / total * 100).round(2).to_dict(),
            'most_common': age_counts.index[0] if len(age_counts) > 0 else 'Unknown',
            'average_age': age_numeric.mean() if len(age_numeric) > 0 else 0,
            'age_std': age_numeric.std() if len(age_numeric) > 0 else 0,
            'diversity_index': self._calculate_diversity_index(age_counts)
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in detections."""
        hourly_counts = df['hour'].value_counts().sort_index()
        daily_counts = df['day_of_week'].value_counts().sort_index()
        
        # Map day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        daily_named = {day_names[day]: count for day, count in daily_counts.items()}
        
        return {
            'hourly_distribution': hourly_counts.to_dict(),
            'daily_distribution': daily_named,
            'peak_hour': hourly_counts.idxmax() if len(hourly_counts) > 0 else 0,
            'peak_day': day_names[daily_counts.idxmax()] if len(daily_counts) > 0 else 'Unknown',
            'busiest_hours': hourly_counts.nlargest(3).to_dict(),
            'quietest_hours': hourly_counts.nsmallest(3).to_dict()
        }
    
    def _analyze_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze confidence scores."""
        return {
            'age_confidence': {
                'mean': df['age_confidence'].mean(),
                'std': df['age_confidence'].std(),
                'min': df['age_confidence'].min(),
                'max': df['age_confidence'].max(),
                'median': df['age_confidence'].median()
            },
            'gender_confidence': {
                'mean': df['gender_confidence'].mean(),
                'std': df['gender_confidence'].std(),
                'min': df['gender_confidence'].min(),
                'max': df['gender_confidence'].max(),
                'median': df['gender_confidence'].median()
            },
            'overall_confidence': {
                'mean': (df['age_confidence'] + df['gender_confidence']).mean() / 2,
                'high_confidence_rate': ((df['age_confidence'] > 0.8) & 
                                       (df['gender_confidence'] > 0.8)).mean() * 100
            }
        }
    
    def _get_peak_hours(self, df: pd.DataFrame) -> Dict:
        """Get peak activity hours."""
        hourly_counts = df['hour'].value_counts().sort_index()
        
        # Define time periods
        morning = hourly_counts[6:12].sum() if len(hourly_counts) > 6 else 0
        afternoon = hourly_counts[12:18].sum() if len(hourly_counts) > 12 else 0
        evening = hourly_counts[18:24].sum() if len(hourly_counts) > 18 else 0
        night = hourly_counts[0:6].sum() + (hourly_counts[24:].sum() if len(hourly_counts) > 24 else 0)
        
        return {
            'morning_6_12': morning,
            'afternoon_12_18': afternoon,
            'evening_18_24': evening,
            'night_0_6': night,
            'peak_period': max([
                ('morning', morning),
                ('afternoon', afternoon),
                ('evening', evening),
                ('night', night)
            ], key=lambda x: x[1])[0]
        }
    
    def _calculate_diversity_index(self, counts: pd.Series) -> float:
        """Calculate Simpson's diversity index."""
        if len(counts) == 0:
            return 0
        
        total = counts.sum()
        proportions = counts / total
        return 1 - (proportions ** 2).sum()
    
    def export_to_csv(self, filename: str = None) -> str:
        """
        Export session data to CSV file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'demographic_data_{timestamp}.csv'
        
        df = pd.DataFrame(self.session_data)
        df.to_csv(filename, index=False)
        
        logger.info(f"Data exported to {filename}")
        return filename
    
    def export_to_json(self, filename: str = None) -> str:
        """
        Export analysis summary to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'demographic_analysis_{timestamp}.json'
        
        summary = self.get_demographic_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Analysis exported to {filename}")
        return filename
    
    def generate_visualizations(self, output_dir: str = 'visualizations') -> List[str]:
        """
        Generate visualization charts and save them.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            List of generated file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.session_data:
            logger.warning("No data available for visualization")
            return []
        
        df = pd.DataFrame(self.session_data)
        generated_files = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Gender distribution pie chart
        plt.figure(figsize=(10, 8))
        gender_counts = df['gender'].value_counts()
        plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        plt.title('Gender Distribution')
        filename = os.path.join(output_dir, 'gender_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        
        # 2. Age distribution bar chart
        plt.figure(figsize=(12, 8))
        age_counts = df['age_group'].value_counts().sort_index()
        plt.bar(age_counts.index, age_counts.values)
        plt.title('Age Group Distribution')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        filename = os.path.join(output_dir, 'age_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        
        # 3. Hourly activity heatmap
        plt.figure(figsize=(15, 8))
        hourly_activity = df.groupby(['hour', 'day_of_week']).size().unstack(fill_value=0)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hourly_activity.columns = [day_names[i] for i in hourly_activity.columns]
        sns.heatmap(hourly_activity.T, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Activity Heatmap (Hour vs Day of Week)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        filename = os.path.join(output_dir, 'activity_heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        
        # 4. Confidence distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(df['age_confidence'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Age Prediction Confidence')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(df['gender_confidence'], bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.title('Gender Prediction Confidence')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        
        # 5. Timeline activity
        plt.figure(figsize=(15, 6))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_resampled = df.set_index('timestamp').resample('10T').size()
        plt.plot(df_resampled.index, df_resampled.values, linewidth=2)
        plt.title('Detection Activity Timeline (10-minute intervals)')
        plt.xlabel('Time')
        plt.ylabel('Number of Detections')
        plt.xticks(rotation=45)
        filename = os.path.join(output_dir, 'timeline_activity.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(filename)
        
        logger.info(f"Generated {len(generated_files)} visualization files")
        return generated_files
    
    def get_real_time_insights(self, window_minutes: int = 30) -> Dict:
        """
        Get insights for the last specified time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary containing recent insights
        """
        if not self.session_data:
            return {'error': 'No data available'}
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_data = [d for d in self.session_data if d['timestamp'] >= cutoff_time]
        
        if not recent_data:
            return {'message': f'No data in the last {window_minutes} minutes'}
        
        df_recent = pd.DataFrame(recent_data)
        
        return {
            'window_minutes': window_minutes,
            'total_detections': len(df_recent),
            'unique_faces': len(df_recent['face_id'].unique()),
            'gender_breakdown': df_recent['gender'].value_counts().to_dict(),
            'age_breakdown': df_recent['age_group'].value_counts().to_dict(),
            'avg_confidence': {
                'age': df_recent['age_confidence'].mean(),
                'gender': df_recent['gender_confidence'].mean()
            },
            'detection_rate': len(df_recent) / window_minutes,  # detections per minute
            'dominant_demographic': self._get_dominant_demographic(df_recent)
        }
    
    def _get_dominant_demographic(self, df: pd.DataFrame) -> Dict:
        """Get the most common demographic combination."""
        if len(df) == 0:
            return {'age_group': 'Unknown', 'gender': 'Unknown', 'count': 0}
        
        combinations = df.groupby(['age_group', 'gender']).size()
        if len(combinations) == 0:
            return {'age_group': 'Unknown', 'gender': 'Unknown', 'count': 0}
        
        dominant = combinations.idxmax()
        count = combinations.max()
        
        return {
            'age_group': dominant[0],
            'gender': dominant[1],
            'count': count,
            'percentage': (count / len(df)) * 100
        }
    
    def clear_data(self):
        """Clear all stored data."""
        self.session_data.clear()
        self.hourly_stats.clear()
        logger.info("Analytics data cleared")
    
    def get_data_size(self) -> Dict:
        """Get information about stored data size."""
        return {
            'total_detections': len(self.session_data),
            'memory_usage_mb': len(str(self.session_data)) / (1024 * 1024),
            'hourly_stats_count': len(self.hourly_stats)
        }