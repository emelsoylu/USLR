import pandas as pd
import numpy as np
from pathlib import Path
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class AdvancedPoseAugmenter:
    def __init__(self, input_csv_path, output_csv_path):
        """
        Advanced pose augmentation with feature engineering for diver sign language
        """
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.df = None
        
        # Problematic classes with insufficient data
        self.problematic_classes = ['show_air', 'okay', 'go_up']
        
        # Keypoint names for pose estimation
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Class-specific signature features for targeted augmentation
        self.class_signature_features = {
            'show_air': {
                'key_points': [9, 10, 7, 8],  # wrists, elbows
                'movement_type': 'horizontal_oscillation',
                'critical_relations': [(9, 10), (7, 8), (5, 6)],  # wrist-wrist, elbow-elbow, shoulder-shoulder
                'amplitude_range': (0.15, 0.35),
                'frequency_range': (0.8, 2.5)
            },
            'okay': {
                'key_points': [9, 10, 0],  # wrists, nose
                'movement_type': 'thumbs_up',
                'critical_relations': [(9, 0), (10, 0)],  # wrist-nose relations
                'height_boost': (0.1, 0.25),
                'confidence_pose': True
            },
            'go_up': {
                'key_points': [9, 10, 7, 8, 0],  # wrists, elbows, nose
                'movement_type': 'upward_directive',
                'critical_relations': [(9, 0), (10, 0), (7, 9), (8, 10)],  # upward chain
                'vertical_emphasis': (0.2, 0.4),
                'pointing_angle': True
            }
        }
        
    def load_data(self):
        """Load and analyze CSV dataset"""
        try:
            self.df = pd.read_csv(self.input_csv_path)
            print(f"INFO: Data loaded successfully: {len(self.df)} samples")
            
            # Class distribution analysis
            class_counts = self.df['class'].value_counts()
            print(f"INFO: Class distribution:\n{class_counts}")
            
            # Analysis of problematic classes
            for class_name in self.problematic_classes:
                count = class_counts.get(class_name, 0)
                print(f"INFO: '{class_name}' - {count} samples")
                
            return True
        except Exception as e:
            print(f"ERROR: Data loading failed: {e}")
            return False
    
    def get_pose_vector(self, row):
        """Extract pose as a vector from row data"""
        coords = []
        for kp_name in self.keypoint_names:
            x = row.get(f'{kp_name}_x', 0.0)
            y = row.get(f'{kp_name}_y', 0.0) 
            z = row.get(f'{kp_name}_z', 0.0)
            vis = row.get(f'{kp_name}_visibility', 0.0)
            coords.extend([x, y, z, vis])
        return np.array(coords)
    
    def set_pose_vector(self, row, pose_vector):
        """Set pose data back to row from vector"""
        for i, kp_name in enumerate(self.keypoint_names):
            base_idx = i * 4
            row[f'{kp_name}_x'] = pose_vector[base_idx]
            row[f'{kp_name}_y'] = pose_vector[base_idx + 1]
            row[f'{kp_name}_z'] = pose_vector[base_idx + 2]
            row[f'{kp_name}_visibility'] = pose_vector[base_idx + 3]
        return row
    
    def extract_signature_features(self, pose_vector, class_name):
        """Extract class-specific signature features"""
        signature = self.class_signature_features[class_name]
        features = {}
        
        # Keypoint positions
        keypoints = []
        for i in range(17):
            x, y, z, vis = pose_vector[i*4:(i+1)*4]
            keypoints.append([x, y, z, vis])
        keypoints = np.array(keypoints)
        
        # Critical relations analysis
        for rel in signature['critical_relations']:
            p1_idx, p2_idx = rel
            if keypoints[p1_idx][3] > 0.1 and keypoints[p2_idx][3] > 0.1:  # Both visible
                # Distance calculation
                dist = np.linalg.norm(keypoints[p1_idx][:3] - keypoints[p2_idx][:3])
                features[f'dist_{p1_idx}_{p2_idx}'] = dist
                
                # Angle calculation
                vec = keypoints[p2_idx][:3] - keypoints[p1_idx][:3]
                angle = np.arctan2(vec[1], vec[0])
                features[f'angle_{p1_idx}_{p2_idx}'] = angle
                
                # Relative height difference
                height_diff = keypoints[p1_idx][1] - keypoints[p2_idx][1]
                features[f'height_{p1_idx}_{p2_idx}'] = height_diff
        
        return features
    
    def signature_based_augmentation(self, row, class_name, intensity=1.0):
        """Advanced signature-based augmentation for specific classes"""
        new_row = row.copy()
        pose_vector = self.get_pose_vector(new_row)
        
        keypoints = []
        for i in range(17):
            x, y, z, vis = pose_vector[i*4:(i+1)*4]
            keypoints.append([x, y, z, vis])
        keypoints = np.array(keypoints)
        
        signature = self.class_signature_features[class_name]
        
        if class_name == 'show_air':
            # Horizontal wave pattern enhancement
            wave_amplitude = np.random.uniform(*signature['amplitude_range']) * intensity
            wave_frequency = np.random.uniform(*signature['frequency_range'])
            wave_phase = np.random.uniform(0, 2*np.pi)
            
            for kp_idx in signature['key_points']:
                if keypoints[kp_idx][3] > 0.1:
                    # Horizontal oscillation
                    wave_offset = wave_amplitude * np.sin(wave_frequency * np.pi + wave_phase)
                    keypoints[kp_idx][0] = np.clip(keypoints[kp_idx][0] + wave_offset, 0.0, 1.0)
                    
                    # Synchronized vertical movement
                    vertical_sync = wave_amplitude * 0.3 * np.cos(wave_frequency * np.pi + wave_phase)
                    keypoints[kp_idx][1] = np.clip(keypoints[kp_idx][1] + vertical_sync, 0.0, 1.0)
                    
                    # Z-depth variation for realism
                    z_variation = np.random.uniform(-0.05, 0.05) * intensity
                    keypoints[kp_idx][2] += z_variation
        
        elif class_name == 'okay':
            # Thumbs up pattern enhancement
            height_boost = np.random.uniform(*signature['height_boost']) * intensity
            
            for kp_idx in signature['key_points']:
                if keypoints[kp_idx][3] > 0.1:
                    if kp_idx in [9, 10]:  # wrists
                        # Strong upward movement for thumbs up
                        keypoints[kp_idx][1] = np.clip(keypoints[kp_idx][1] - height_boost, 0.0, 1.0)
                        
                        # Forward Z movement (confidence gesture)
                        z_forward = np.random.uniform(0.05, 0.15) * intensity
                        keypoints[kp_idx][2] += z_forward
                        
                        # Slight inward rotation for thumb position
                        rotation_factor = 0.05 * intensity
                        center_x = 0.5  # Assume body center
                        if kp_idx == 9:  # left wrist
                            keypoints[kp_idx][0] = np.clip(keypoints[kp_idx][0] + rotation_factor, 0.0, 1.0)
                        else:  # right wrist
                            keypoints[kp_idx][0] = np.clip(keypoints[kp_idx][0] - rotation_factor, 0.0, 1.0)
        
        elif class_name == 'go_up':
            # Upward directive pattern enhancement
            vertical_boost = np.random.uniform(*signature['vertical_emphasis']) * intensity
            
            # Create upward pointing gesture
            for kp_idx in signature['key_points']:
                if keypoints[kp_idx][3] > 0.1:
                    if kp_idx in [9, 10]:  # wrists - main pointing
                        keypoints[kp_idx][1] = np.clip(keypoints[kp_idx][1] - vertical_boost, 0.0, 1.0)
                        
                        # Point towards nose/head area
                        if keypoints[0][3] > 0.1:  # nose visible
                            direction_x = keypoints[0][0] - keypoints[kp_idx][0]
                            direction_y = keypoints[0][1] - keypoints[kp_idx][1]
                            
                            # Move slightly towards pointing direction
                            pointing_strength = 0.1 * intensity
                            keypoints[kp_idx][0] += direction_x * pointing_strength
                            keypoints[kp_idx][1] += direction_y * pointing_strength
                            
                            # Clip to valid range
                            keypoints[kp_idx][0] = np.clip(keypoints[kp_idx][0], 0.0, 1.0)
                            keypoints[kp_idx][1] = np.clip(keypoints[kp_idx][1], 0.0, 1.0)
                    
                    elif kp_idx in [7, 8]:  # elbows - supporting movement
                        supporting_boost = vertical_boost * 0.6
                        keypoints[kp_idx][1] = np.clip(keypoints[kp_idx][1] - supporting_boost, 0.0, 1.0)
        
        # Update pose vector
        new_pose_vector = keypoints.flatten()
        return self.set_pose_vector(new_row, new_pose_vector)
    
    def adversarial_augmentation(self, row, class_name):
        """Adversarial-style augmentation to create challenging samples"""
        new_row = row.copy()
        pose_vector = self.get_pose_vector(new_row)
        
        keypoints = []
        for i in range(17):
            x, y, z, vis = pose_vector[i*4:(i+1)*4]
            keypoints.append([x, y, z, vis])
        keypoints = np.array(keypoints)
        
        # Add confusing elements to make classification more challenging
        if class_name == 'show_air':
            # Add vertical movements that might confuse with 'go_up'
            for kp_idx in [9, 10]:
                if keypoints[kp_idx][3] > 0.1:
                    confusing_vertical = np.random.uniform(-0.1, 0.1)
                    keypoints[kp_idx][1] = np.clip(keypoints[kp_idx][1] + confusing_vertical, 0.0, 1.0)
        
        elif class_name == 'okay':
            # Add horizontal movements that might confuse with 'show_air'
            for kp_idx in [9, 10]:
                if keypoints[kp_idx][3] > 0.1:
                    confusing_horizontal = np.random.uniform(-0.08, 0.08)
                    keypoints[kp_idx][0] = np.clip(keypoints[kp_idx][0] + confusing_horizontal, 0.0, 1.0)
        
        elif class_name == 'go_up':
            # Add thumb-like positions that might confuse with 'okay'
            for kp_idx in [9, 10]:
                if keypoints[kp_idx][3] > 0.1:
                    # Sometimes add slight downward movement
                    if np.random.random() < 0.3:
                        confusing_down = np.random.uniform(0.05, 0.15)
                        keypoints[kp_idx][1] = np.clip(keypoints[kp_idx][1] + confusing_down, 0.0, 1.0)
        
        # Update pose vector
        new_pose_vector = keypoints.flatten()
        return self.set_pose_vector(new_row, new_pose_vector)
    
    def interpolation_augmentation(self, class_data, n_samples=100):
        """Generate new samples through interpolation within class"""
        if len(class_data) < 2:
            return []
        
        augmented_samples = []
        
        for _ in range(n_samples):
            # Select two random samples
            sample1 = class_data.sample(n=1).iloc[0]
            sample2 = class_data.sample(n=1).iloc[0]
            
            pose1 = self.get_pose_vector(sample1)
            pose2 = self.get_pose_vector(sample2)
            
            # Interpolation weight using Beta distribution for more middle values
            alpha = np.random.beta(2, 2)
            
            # Interpolate between poses
            interpolated_pose = alpha * pose1 + (1 - alpha) * pose2
            
            # Create new row
            new_row = sample1.copy()
            new_row = self.set_pose_vector(new_row, interpolated_pose)
            
            augmented_samples.append(new_row)
        
        return augmented_samples
    
    def manifold_augmentation(self, class_data, n_samples=100):
        """Manifold learning-based augmentation"""
        if len(class_data) < 10:
            return []
        
        # Extract pose vectors
        pose_vectors = []
        for idx, row in class_data.iterrows():
            pose_vector = self.get_pose_vector(row)
            pose_vectors.append(pose_vector)
        
        pose_vectors = np.array(pose_vectors)
        
        try:
            # PCA for dimensionality reduction
            pca = PCA(n_components=min(10, len(pose_vectors)-1))
            reduced_poses = pca.fit_transform(pose_vectors)
            
            # Generate new points in reduced space
            augmented_samples = []
            
            for _ in range(n_samples):
                # Gaussian noise in reduced space
                noise = np.random.normal(0, 0.1, reduced_poses.shape[1])
                
                # Random base sample
                base_idx = np.random.randint(len(reduced_poses))
                base_reduced = reduced_poses[base_idx]
                
                # Add noise
                new_reduced = base_reduced + noise
                
                # Transform back to original space
                new_pose_vector = pca.inverse_transform(new_reduced.reshape(1, -1))[0]
                
                # Clip to valid ranges
                for i in range(0, len(new_pose_vector), 4):
                    # x, y coordinates clip to [0, 1]
                    new_pose_vector[i] = np.clip(new_pose_vector[i], 0.0, 1.0)
                    new_pose_vector[i+1] = np.clip(new_pose_vector[i+1], 0.0, 1.0)
                    # z can be any value
                    # visibility clip to [0, 1]
                    new_pose_vector[i+3] = np.clip(new_pose_vector[i+3], 0.0, 1.0)
                
                # Create new row
                base_row = class_data.iloc[base_idx].copy()
                new_row = self.set_pose_vector(base_row, new_pose_vector)
                augmented_samples.append(new_row)
            
            return augmented_samples
            
        except Exception as e:
            print(f"WARNING: Manifold augmentation failed: {e}")
            return []
    
    def create_advanced_samples(self, class_name, target_count=1000):
        """Generate advanced augmented samples for specific class"""
        class_data = self.df[self.df['class'] == class_name].copy()
        
        if len(class_data) == 0:
            print(f"WARNING: No data found for class '{class_name}'")
            return pd.DataFrame()
        
        print(f"INFO: Starting advanced augmentation for '{class_name}'")
        print(f"INFO: Original sample count: {len(class_data)}, Target: {target_count}")
        
        all_augmented = []
        
        # 1. Signature-based augmentation (40% of target)
        signature_count = int(target_count * 0.4)
        print(f"INFO: Signature-based: {signature_count} samples")
        
        for intensity in [0.5, 1.0, 1.5, 2.0]:  # Different intensities
            count_per_intensity = signature_count // 4
            for _ in range(count_per_intensity):
                original_row = class_data.sample(n=1).iloc[0]
                augmented_row = self.signature_based_augmentation(original_row, class_name, intensity)
                all_augmented.append(augmented_row)
        
        # 2. Adversarial augmentation (20% of target)
        adversarial_count = int(target_count * 0.2)
        print(f"INFO: Adversarial: {adversarial_count} samples")
        
        for _ in range(adversarial_count):
            original_row = class_data.sample(n=1).iloc[0]
            augmented_row = self.adversarial_augmentation(original_row, class_name)
            all_augmented.append(augmented_row)
        
        # 3. Interpolation augmentation (25% of target)
        interpolation_count = int(target_count * 0.25)
        print(f"INFO: Interpolation: {interpolation_count} samples")
        
        interpolated_samples = self.interpolation_augmentation(class_data, interpolation_count)
        all_augmented.extend(interpolated_samples)
        
        # 4. Manifold augmentation (15% of target)
        manifold_count = int(target_count * 0.15)
        print(f"INFO: Manifold: {manifold_count} samples")
        
        manifold_samples = self.manifold_augmentation(class_data, manifold_count)
        all_augmented.extend(manifold_samples)
        
        # Fill remaining samples with best technique
        remaining = target_count - len(all_augmented)
        if remaining > 0:
            print(f"INFO: Filling remaining {remaining} with best technique")
            for _ in range(remaining):
                original_row = class_data.sample(n=1).iloc[0]
                augmented_row = self.signature_based_augmentation(original_row, class_name, 1.5)
                all_augmented.append(augmented_row)
        
        print(f"INFO: Total {len(all_augmented)} advanced samples created for '{class_name}'")
        
        return pd.DataFrame(all_augmented)
    
    def generate_advanced_dataset(self, target_samples_per_class=1000):
        """Generate advanced augmented dataset"""
        if self.df is None:
            print("ERROR: Data must be loaded first!")
            return False
        
        all_augmented_data = []
        
        # Apply advanced augmentation for each problematic class
        for class_name in self.problematic_classes:
            advanced_samples = self.create_advanced_samples(class_name, target_samples_per_class)
            if not advanced_samples.empty:
                all_augmented_data.append(advanced_samples)
        
        if not all_augmented_data:
            print("ERROR: No advanced data could be generated!")
            return False
        
        # Combine all augmented data
        final_augmented_df = pd.concat(all_augmented_data, ignore_index=True)
        
        # Save to CSV
        final_augmented_df.to_csv(self.output_csv_path, index=False)
        
        print(f"INFO: Advanced augmented dataset saved: {self.output_csv_path}")
        print(f"INFO: Total advanced sample count: {len(final_augmented_df)}")
        print(f"INFO: Advanced class distribution:\n{final_augmented_df['class'].value_counts()}")
        
        return True
    
    def combine_with_existing(self, existing_csv_path, combined_output_path):
        """Combine existing dataset with advanced augmented dataset"""
        try:
            # Load existing data
            existing_df = pd.read_csv(existing_csv_path)
            
            # Load advanced augmented data
            advanced_df = pd.read_csv(self.output_csv_path)
            
            # Combine datasets
            combined_df = pd.concat([existing_df, advanced_df], ignore_index=True)
            
            # Save combined dataset
            combined_df.to_csv(combined_output_path, index=False)
            
            print(f"INFO: Datasets combined successfully: {combined_output_path}")
            print(f"INFO: Existing sample count: {len(existing_df)}")
            print(f"INFO: Advanced sample count: {len(advanced_df)}")
            print(f"INFO: Total sample count: {len(combined_df)}")
            print(f"INFO: Final class distribution:\n{combined_df['class'].value_counts()}")
            
            # Final status of problematic classes
            for class_name in self.problematic_classes:
                count = combined_df['class'].value_counts().get(class_name, 0)
                print(f"INFO: '{class_name}' final count: {count}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Data combination failed: {e}")
            return False

def main():
    """Main execution function"""
    # Configuration parameters
    input_csv_path = "pose_features_3d_advanced_combined.csv"  # Your latest dataset
    advanced_csv_path = "pose_features_3d_advanced.csv"  # Advanced augmented data only
    ultimate_csv_path = "pose_features_3d_ultimate.csv"  # All data combined
    
    # Create Advanced Pose Augmenter
    augmenter = AdvancedPoseAugmenter(input_csv_path, advanced_csv_path)
    
    # Load data
    if not augmenter.load_data():
        return
    
    # Generate advanced augmented dataset
    # 1000 new samples per problematic class (TOTAL 3000 NEW SAMPLES!)
    if augmenter.generate_advanced_dataset(target_samples_per_class=1000):
        print("INFO: 🔥 Advanced augmentation completed successfully!")
        
        # Combine with existing data
        if augmenter.combine_with_existing(input_csv_path, ultimate_csv_path):
            print("INFO: ✅ All operations completed successfully!")
            print(f"INFO: 🎯 ULTIMATE dataset ready: {ultimate_csv_path}")
            print(f"INFO: Retrain your model with this dataset!")
        else:
            print("ERROR: Data combination failed!")
    else:
        print("ERROR: Advanced augmentation failed!")

if __name__ == "__main__":
    main()