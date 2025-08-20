import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import deque
import pickle

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplePose3DLifter(nn.Module):
    """Simple 3D pose lifting network with visibility prediction"""
    def __init__(self, input_dim=51, hidden_dim=256):  # Base input: 34 coords + 17 visibility = 51
        super().__init__()
        # Dynamic input size calculation
        # Base: 51 (34 coords + 17 visibility)
        # With temporal: +34 (velocity) +34 (acceleration) +17 (visibility_change) = 136 total
        max_input_dim = 136  # Maximum possible input size with all temporal features
        
        self.coordinate_network = nn.Sequential(
            nn.Linear(max_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 51)  # 17 keypoints * 3 coordinates
        )
        
        self.visibility_network = nn.Sequential(
            nn.Linear(max_input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 17),  # 17 visibility scores
            nn.Sigmoid()  # Visibility between 0-1
        )
        
    def forward(self, x):
        # Pad input to maximum size if needed
        if x.size(1) < 136:
            batch_size = x.size(0)
            padding = torch.zeros(batch_size, 136 - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
            
        coords_3d = self.coordinate_network(x)
        visibility_3d = self.visibility_network(x)
        return coords_3d, visibility_3d

class TemporalPoseProcessor:
    """Temporal context processing for pose sequences"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.pose_history = deque(maxlen=window_size)
        
    def add_pose(self, pose_2d, visibility_2d):
        """Add new pose and visibility data"""
        pose_data = {
            'coordinates': pose_2d,
            'visibility': visibility_2d
        }
        self.pose_history.append(pose_data)
        
    def get_temporal_features(self):
        """Extract temporal features from pose sequence"""
        if len(self.pose_history) < 2:
            return None
            
        # Calculate velocity (frame-to-frame difference)
        current_pose = np.array(self.pose_history[-1]['coordinates'])
        prev_pose = np.array(self.pose_history[-2]['coordinates'])
        velocity = current_pose - prev_pose
        
        # Visibility changes
        current_vis = np.array(self.pose_history[-1]['visibility'])
        prev_vis = np.array(self.pose_history[-2]['visibility'])
        visibility_change = current_vis - prev_vis
        
        # Calculate acceleration (if enough frames available)
        if len(self.pose_history) >= 3:
            prev_prev_pose = np.array(self.pose_history[-3]['coordinates'])
            prev_velocity = prev_pose - prev_prev_pose
            acceleration = velocity - prev_velocity
        else:
            acceleration = np.zeros_like(velocity)
            
        return {
            'pose': current_pose,
            'velocity': velocity,
            'acceleration': acceleration,
            'visibility': current_vis,
            'visibility_change': visibility_change
        }
        
    def smooth_pose(self):
        """Apply pose smoothing algorithm"""
        if len(self.pose_history) < 3:
            if self.pose_history:
                return self.pose_history[-1]['coordinates'], self.pose_history[-1]['visibility']
            else:
                return None, None
            
        # Gaussian smoothing for coordinates
        poses = np.array([item['coordinates'] for item in self.pose_history])
        visibilities = np.array([item['visibility'] for item in self.pose_history])
        
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])[:len(poses)]
        weights = weights / weights.sum()
        
        smoothed_pose = np.average(poses, axis=0, weights=weights)
        smoothed_visibility = np.average(visibilities, axis=0, weights=weights)
        
        return smoothed_pose.tolist(), smoothed_visibility.tolist()

class YOLO11Pose3DExtractor:
    def __init__(self, data_path, target_frames=150, output_csv='pose_features_3d.csv', 
                 use_pretrained_lifter=True):
        """
        YOLO11-pose + VideoPose3D for 3D feature extraction
        
        Args:
            data_path (str): Main directory containing video folders
            target_frames (int): Number of frames to extract from each video
            output_csv (str): Output CSV filename
            use_pretrained_lifter (bool): Use pretrained lifter weights
        """
        self.data_path = Path(data_path)
        self.target_frames = target_frames
        self.output_csv = output_csv
        self.use_pretrained_lifter = use_pretrained_lifter
        
        self.yolo_model = None
        self.pose_lifter = None
        self.features_data = []
        
        # Supported video formats
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_models(self):
        """Load YOLO11-pose and 3D lifter models"""
        try:
            # Load YOLO11-pose model
            self.yolo_model = YOLO('yolo11n-pose.pt')
            logger.info("YOLO11-pose model loaded successfully")
            
            # Load 3D Pose Lifter model
            self.pose_lifter = SimplePose3DLifter().to(self.device)
            
            if self.use_pretrained_lifter:
                # Try to load pretrained weights
                try:
                    checkpoint_path = 'pose_lifter_weights.pth'
                    if os.path.exists(checkpoint_path):
                        self.pose_lifter.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                        logger.info("Pretrained 3D lifter weights loaded")
                    else:
                        logger.warning("Pretrained weights not found, using random initialization")
                        self._initialize_lifter_weights()
                except Exception as e:
                    logger.warning(f"Weight loading error: {e}, using random initialization")
                    self._initialize_lifter_weights()
            else:
                self._initialize_lifter_weights()
                
            self.pose_lifter.eval()
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise
    
    def _initialize_lifter_weights(self):
        """Initialize weights for 3D lifter"""
        for m in self.pose_lifter.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def calculate_step_size(self, total_frames):
        """Calculate step size based on target frame count"""
        if total_frames >= self.target_frames:
            step_size = total_frames / self.target_frames
            return step_size, 1
        else:
            repeat_count = int(np.ceil(self.target_frames / total_frames))
            return 1, repeat_count
    
    def extract_2d_pose(self, frame):
        """Extract 2D pose with visibility scores"""
        try:
            results = self.yolo_model(frame, verbose=False)
            
            best_detection = None
            best_visibilities = None
            best_confidence = 0
            
            for result in results:
                if result.keypoints is not None:
                    for keypoints in result.keypoints.data:
                        confidences = keypoints[:, 2]  # YOLO confidence scores
                        avg_confidence = confidences.mean().item()
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_detection = keypoints[:, :2]  # x, y coordinates
                            best_visibilities = keypoints[:, 2]  # Use YOLO confidence as visibility
            
            if best_detection is not None:
                # Normalize coordinates
                h, w = frame.shape[:2]
                normalized_keypoints = []
                normalized_visibilities = []
                
                for i, keypoint in enumerate(best_detection):
                    x, y = keypoint
                    normalized_x = x.item() / w if w > 0 else 0
                    normalized_y = y.item() / h if h > 0 else 0
                    visibility = best_visibilities[i].item()
                    
                    normalized_keypoints.extend([normalized_x, normalized_y])
                    normalized_visibilities.append(visibility)
                
                return normalized_keypoints, normalized_visibilities, best_confidence
            else:
                return [0.0] * 34, [0.0] * 17, 0.0
                
        except Exception as e:
            logger.warning(f"2D pose extraction error: {e}")
            return [0.0] * 34, [0.0] * 17, 0.0
    
    def lift_to_3d(self, pose_2d, visibility_2d, temporal_features=None):
        """Lift 2D pose to 3D with visibility prediction"""
        try:
            if all(x == 0.0 for x in pose_2d):
                return [0.0] * 51, [0.0] * 17
            
            # Prepare input tensor (2D coordinates + visibility)
            input_features = pose_2d + visibility_2d  # 34 + 17 = 51
            
            # Add temporal features if available
            if temporal_features is not None:
                input_features.extend(temporal_features['velocity'])      # +34 = 85
                input_features.extend(temporal_features['acceleration'])  # +34 = 119
                input_features.extend(temporal_features['visibility_change'])  # +17 = 136
            
            pose_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
            
            # 3D prediction
            with torch.no_grad():
                pose_3d, visibility_3d = self.pose_lifter(pose_tensor)
                pose_3d = pose_3d.cpu().numpy().flatten()
                visibility_3d = visibility_3d.cpu().numpy().flatten()
            
            return pose_3d.tolist(), visibility_3d.tolist()
            
        except Exception as e:
            logger.warning(f"3D lifting error: {e}")
            return [0.0] * 51, [0.0] * 17
    
    def extract_pose_features_3d(self, frame, temporal_processor):
        """Extract 3D pose features with visibility"""
        try:
            # 2D pose detection
            pose_2d, visibility_2d, confidence = self.extract_2d_pose(frame)
            
            # Add to temporal processor
            temporal_processor.add_pose(pose_2d, visibility_2d)
            
            # Get temporal features
            temporal_features = temporal_processor.get_temporal_features()
            
            # Get smoothed pose
            smoothed_pose, smoothed_visibility = temporal_processor.smooth_pose()
            if smoothed_pose is None:
                smoothed_pose = pose_2d
                smoothed_visibility = visibility_2d
            
            # Lift to 3D
            pose_3d, visibility_3d = self.lift_to_3d(smoothed_pose, smoothed_visibility, temporal_features)
            
            # Combine 3D coordinates and visibility
            combined_features = []
            for i in range(17):  # 17 keypoints
                x_idx, y_idx, z_idx = i*3, i*3+1, i*3+2
                vis_idx = i
                combined_features.extend([
                    pose_3d[x_idx], pose_3d[y_idx], pose_3d[z_idx], visibility_3d[vis_idx]
                ])
            
            # Add overall confidence
            combined_features.append(confidence)
            
            return combined_features
            
        except Exception as e:
            logger.warning(f"3D pose feature extraction error: {e}")
            return [0.0] * 69  # 17 keypoints * 4 (x,y,z,visibility) + 1 confidence
    
    
    def process_video(self, video_path, class_name):
        """Process single video and extract 3D features"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.warning(f"Cannot read video or empty: {video_path}")
                cap.release()
                return []
            
            logger.info(f"Processing: {video_path.name} (Total frames: {total_frames})")
            
            # Calculate step size
            step_size, repeat_count = self.calculate_step_size(total_frames)
            
            # Initialize temporal processor
            temporal_processor = TemporalPoseProcessor(window_size=5)
            
            video_features = []
            frame_indices = []
            
            if total_frames >= self.target_frames:
                frame_indices = [int(i * step_size) for i in range(self.target_frames)]
            else:
                base_indices = list(range(total_frames))
                frame_indices = (base_indices * repeat_count)[:self.target_frames]
            
            # Process frames
            for frame_idx in tqdm(frame_indices, desc=f"Processing {video_path.name}", leave=False):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Cannot read frame {frame_idx}")
                    video_features.append([0.0] * 69)
                    continue
                
                # Extract 3D pose features
                pose_features_3d = self.extract_pose_features_3d(frame, temporal_processor)
                video_features.append(pose_features_3d)
            
            cap.release()
            
            # Add class name for each frame
            processed_features = []
            for features in video_features:
                row = [class_name] + features
                processed_features.append(row)
            
            return processed_features
            
        except Exception as e:
            logger.error(f"Video processing error {video_path}: {e}")
            return []
    
    def scan_videos(self):
        """Scan all videos in class directories"""
        video_files = []
        
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                for file_path in class_dir.iterdir():
                    if file_path.suffix.lower() in self.video_extensions:
                        video_files.append((file_path, class_name))
        
        logger.info(f"Found {len(video_files)} video files")
        return video_files
    
    def create_column_names(self):
        """Create column names for CSV (3D + visibility)"""
        columns = ['class']
        
        # COCO pose keypoint names
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Add x, y, z, visibility for each keypoint
        for keypoint in keypoint_names:
            columns.extend([f'{keypoint}_x', f'{keypoint}_y', f'{keypoint}_z', f'{keypoint}_visibility'])
        
        # Add overall confidence score
        columns.append('confidence')
        
        return columns
    
    def extract_features(self):
        """Main 3D feature extraction function"""
        logger.info("Starting YOLO11-pose + VideoPose3D feature extraction...")
        
        # Load models
        self.load_models()
        
        # Scan all videos
        video_files = self.scan_videos()
        
        if not video_files:
            logger.error("No video files found!")
            return
        
        # Process each video
        all_features = []
        
        for video_path, class_name in tqdm(video_files, desc="Processing videos"):
            video_features = self.process_video(video_path, class_name)
            all_features.extend(video_features)
        
        if not all_features:
            logger.error("No features extracted!")
            return
        
        # Create DataFrame and save to CSV
        columns = self.create_column_names()
        df = pd.DataFrame(all_features, columns=columns)
        
        # Save to CSV
        df.to_csv(self.output_csv, index=False)
        logger.info(f"3D features (with visibility) saved successfully: {self.output_csv}")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Feature dimension: {len(columns)-1} (68 values: 17 keypoints × 4 features + 1 confidence)")
        logger.info(f"Class distribution:\n{df['class'].value_counts()}")
        
        # Statistics
        logger.info(f"Average confidence: {df['confidence'].mean():.3f}")
        logger.info(f"Minimum confidence: {df['confidence'].min():.3f}")
        
        # Visibility statistics
        visibility_columns = [col for col in df.columns if 'visibility' in col]
        if visibility_columns:
            avg_visibility = df[visibility_columns].mean().mean()
            logger.info(f"Average visibility: {avg_visibility:.3f}")
            
            # Show keypoints with lowest visibility
            min_visibility_per_keypoint = df[visibility_columns].mean().sort_values()
            logger.info(f"Keypoints with lowest visibility:\n{min_visibility_per_keypoint.head()}")
        
    def save_lifter_weights(self, path='pose_lifter_weights.pth'):
        """Save 3D lifter weights"""
        if self.pose_lifter is not None:
            torch.save(self.pose_lifter.state_dict(), path)
            logger.info(f"Lifter weights saved: {path}")

def main():
    """Main execution function"""
    # Parameters
    data_path = r"D:\diver_language\data"  # Main directory containing video folders
    target_frames = 150  # Number of frames to extract from each video
    output_csv = "pose_features_3d.csv"  # Output CSV file
    
    # Create and run 3D feature extractor
    extractor = YOLO11Pose3DExtractor(
        data_path=data_path, 
        target_frames=target_frames, 
        output_csv=output_csv,
        use_pretrained_lifter=True
    )
    
    extractor.extract_features()
    
    # Save weights for future use
    extractor.save_lifter_weights()

if __name__ == "__main__":
    main()