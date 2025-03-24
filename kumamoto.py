# Install necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import laspy
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from scipy.stats import pearsonr
from scipy import ndimage
from scipy.sparse import csr_matrix
import gc  # Garbage collector
import warnings
warnings.filterwarnings('ignore')

# File paths for the pre and post earthquake point cloud data
PRE_LAZ_PATH = "PCR Disaster recovery/data/pretry2.laz"
POST_LAZ_PATH = "PCR Disaster recovery/data/posttry2.laz"

def load_laz_file(file_path):
    """Load LAZ file and return as numpy array"""
    print(f"Loading {file_path}...")
    start_time = time.time()
    
    try:
        # Try with explicit backend
        from laspy.compression import LazBackend
        las = laspy.read(file_path, laz_backend=LazBackend.LazrsBackend)
    except:
        try:
            # Try with laszip backend
            las = laspy.read(file_path, laz_backend=LazBackend.LaszipBackend)
        except:
            # Fallback to default
            las = laspy.read(file_path)
        
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    print(f"Loaded {len(points)} points in {time.time() - start_time:.2f} seconds")
    return points, las

def create_dsm(points, resolution=0.5, x_range=None, y_range=None):
    """Create a Digital Surface Model from point cloud - memory optimized"""
    if x_range is None:
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    else:
        x_min, x_max = x_range
        
    if y_range is None:
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    else:
        y_min, y_max = y_range
    
    # Create grid
    x_size = int((x_max - x_min) / resolution) + 1
    y_size = int((y_max - y_min) / resolution) + 1
    
    print(f"Creating DSM with dimensions: {y_size} x {x_size}")
    
    # Initialize DSM with NaN values
    dsm = np.full((y_size, x_size), np.nan)
    
    # Process in smaller batches to reduce memory usage
    batch_size = 250000  # Smaller batch size
    num_batches = (len(points) + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        if b % 10 == 0:
            print(f"Processing batch {b+1}/{num_batches}")
            
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(points))
        batch_points = points[start_idx:end_idx]
        
        # Calculate indices for this batch
        x_indices = np.floor((batch_points[:, 0] - x_min) / resolution).astype(int)
        y_indices = np.floor((batch_points[:, 1] - y_min) / resolution).astype(int)
        
        # Filter valid indices
        valid_mask = (x_indices >= 0) & (x_indices < x_size) & (y_indices >= 0) & (y_indices < y_size)
        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        z_values = batch_points[valid_mask, 2]
        
        # Update DSM with maximum height for each cell
        for i in range(len(x_indices)):
            if np.isnan(dsm[y_indices[i], x_indices[i]]) or z_values[i] > dsm[y_indices[i], x_indices[i]]:
                dsm[y_indices[i], x_indices[i]] = z_values[i]
        
        # Free memory
        del batch_points, x_indices, y_indices, z_values, valid_mask
        gc.collect()
    
    return dsm, (x_min, y_min, resolution)

def load_and_process_data(pre_file, post_file, resolution=1.0):
    """Load and process both pre and post-earthquake data"""
    # Load point clouds
    pre_points, pre_las = load_laz_file(pre_file)
    post_points, post_las = load_laz_file(post_file)
    
    # Find common bounds
    x_min = max(np.min(pre_points[:, 0]), np.min(post_points[:, 0]))
    x_max = min(np.max(pre_points[:, 0]), np.max(post_points[:, 0]))
    y_min = max(np.min(pre_points[:, 1]), np.min(post_points[:, 1]))
    y_max = min(np.max(pre_points[:, 1]), np.max(post_points[:, 1]))
    
    print(f"Common bounds: X({x_min:.2f}, {x_max:.2f}), Y({y_min:.2f}, {y_max:.2f})")
    
    # Create DSMs with common bounds
    pre_dsm, dsm_params = create_dsm(pre_points, resolution, (x_min, x_max), (y_min, y_max))
    post_dsm, _ = create_dsm(post_points, resolution, (x_min, x_max), (y_min, y_max))
    
    return pre_dsm, post_dsm, dsm_params, pre_points, post_points

def simulate_building_footprints(pre_dsm, post_dsm, min_building_size=20, height_threshold=1.8, max_buildings=8000):
    """
    Memory-optimized version of building footprint extraction
    Process in chunks to avoid memory issues
    Parameters adjusted to detect more buildings
    """
    print("Extracting building footprints...")
    start_time = time.time()
    
    # Identify potential buildings (areas elevated above ground)
    ground_level = np.nanpercentile(pre_dsm, 10)
    print(f"Estimated ground level: {ground_level:.2f} m")
    
    # Find areas that are elevated above ground level (reduced threshold to detect more buildings)
    building_mask = (pre_dsm - ground_level) > height_threshold
    
    # Process in smaller chunks to reduce memory usage
    chunk_size = 500  # Smaller chunk size
    height, width = building_mask.shape
    
    # Create a structure for 8-connectivity
    s = ndimage.generate_binary_structure(2, 2)
    
    # Initialize with zeros
    labeled_buildings = np.zeros_like(building_mask, dtype=np.int32)
    current_label = 1
    
    print(f"Processing building mask in chunks...")
    
    # Process the mask in smaller chunks
    for i in range(0, height, chunk_size):
        end_i = min(i + chunk_size, height)
        
        for j in range(0, width, chunk_size):
            end_j = min(j + chunk_size, width)
            
            # Extract chunk
            chunk_mask = building_mask[i:end_i, j:end_j].copy()
            if not np.any(chunk_mask):
                continue
                
            # Label the chunk
            chunk_labels, _ = ndimage.label(chunk_mask, structure=s)
            
            # Skip if no buildings in this chunk
            if np.max(chunk_labels) == 0:
                continue
                
            # Process each label in the chunk
            for label in range(1, np.max(chunk_labels) + 1):
                footprint = chunk_labels == label
                if np.sum(footprint) >= min_building_size:
                    # Add to the global labeled image
                    labeled_buildings[i:end_i, j:end_j][footprint] = current_label
                    current_label += 1
            
            # Free memory
            del chunk_mask, chunk_labels
            gc.collect()
    
    print(f"Labeling completed. Processing footprints...")
    
    # Extract footprints more efficiently - limit to max_buildings
    max_buildings = min(max_buildings, np.max(labeled_buildings))
    footprints = []
    
    # Process in smaller batches
    batch_size = 50  # Very small batch size
    num_batches = (max_buildings + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        start_label = b * batch_size + 1
        end_label = min((b + 1) * batch_size, max_buildings) + 1
        
        if b % 5 == 0:
            print(f"Processing building batch {b+1}/{num_batches}")
        
        for label in range(start_label, end_label):
            footprint = labeled_buildings == label
            if np.sum(footprint) >= min_building_size:
                # Store as sparse matrix to save memory
                footprints.append(csr_matrix(footprint))
        
        # Force garbage collection
        gc.collect()
    
    print(f"Identified {len(footprints)} potential buildings in {time.time() - start_time:.2f} seconds")
    return footprints

def calculate_building_features(pre_dsm, post_dsm, footprints):
    """Calculate features for each building - extreme memory optimization"""
    print("Calculating building features...")
    start_time = time.time()
    
    features = []
    batch_size = 20  # Very small batch size
    num_batches = (len(footprints) + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(footprints))
        
        if b % 10 == 0:
            print(f"Processing building batch {b+1}/{num_batches}")
        
        for i in range(start_idx, end_idx):
            # Convert sparse matrix to dense boolean array for this building only
            footprint = footprints[i].toarray().astype(bool)
            
            # Extract height values for this building
            pre_heights = pre_dsm[footprint]
            post_heights = post_dsm[footprint]
            
            # Skip if no valid data
            if np.sum(~np.isnan(pre_heights)) == 0 or np.sum(~np.isnan(post_heights)) == 0:
                continue
            
            # Calculate features
            delta_h = np.nanmean(post_heights - pre_heights)
            std_dev = np.nanstd(post_heights - pre_heights)
            
            # Correlation coefficient
            valid_mask = ~np.isnan(pre_heights) & ~np.isnan(post_heights)
            if np.sum(valid_mask) < 2:  # Need at least 2 points for correlation
                continue
                
            pre_valid = pre_heights[valid_mask]
            post_valid = post_heights[valid_mask]
            
            if len(pre_valid) > 0 and len(post_valid) > 0:
                try:
                    corr, _ = pearsonr(pre_valid, post_valid)
                except:
                    corr = 0
            else:
                corr = 0
                
            # Calculate building centroid for visualization
            rows, cols = np.where(footprint)
            centroid_y = np.mean(rows)
            centroid_x = np.mean(cols)
            
            # Calculate additional features to improve classification
            max_height_diff = np.nanmax(np.abs(post_heights - pre_heights))
            min_height_diff = np.nanmin(post_heights - pre_heights)
            height_range = np.nanmax(pre_heights) - np.nanmin(pre_heights)
            
            # Store features and metadata
            features.append({
                'delta_h': delta_h,
                'std_dev': std_dev,
                'correlation': corr,
                'max_height_diff': max_height_diff,
                'min_height_diff': min_height_diff,
                'height_range': height_range,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'footprint': footprint,
                'area': np.sum(footprint)
            })
            
            # Free memory
            del footprint, pre_heights, post_heights, valid_mask
            if 'pre_valid' in locals(): del pre_valid
            if 'post_valid' in locals(): del post_valid
        
        # Free memory for this batch
        gc.collect()
    
    print(f"Calculated features for {len(features)} buildings in {time.time() - start_time:.2f} seconds")
    return features

def detect_collapsed_buildings(features, threshold=-0.5):
    """Detect collapsed buildings based on height difference threshold"""
    collapsed = []
    non_collapsed = []
    
    for building in features:
        if building['delta_h'] < threshold:
            building['status'] = 'collapsed'
            collapsed.append(building)
        else:
            building['status'] = 'non_collapsed'
            non_collapsed.append(building)
    
    print(f"Detected {len(collapsed)} collapsed buildings and {len(non_collapsed)} non-collapsed buildings")
    return collapsed, non_collapsed

def visualize_dsm_difference(pre_dsm, post_dsm, dsm_params, collapsed_buildings, non_collapsed_buildings):
    """Visualize the difference between pre and post-earthquake DSMs"""
    print("Creating DSM difference visualization...")
    plt.figure(figsize=(16, 14))
    
    # Calculate height difference
    diff = post_dsm - pre_dsm
    
    # Create a masked array for better visualization
    masked_diff = np.ma.masked_invalid(diff)
    
    # Plot the difference
    cmap = plt.cm.RdBu_r
    cmap.set_bad('white', 1.)
    
    plt.imshow(masked_diff, cmap=cmap, vmin=-5, vmax=5)
    plt.colorbar(label='Height Difference (m)')
    
    # Plot building centroids (limit number to reduce memory usage)
    max_buildings_to_plot = 1000
    
    for building in collapsed_buildings[:min(len(collapsed_buildings), max_buildings_to_plot)]:
        plt.plot(building['centroid_x'], building['centroid_y'], 'rx', markersize=4)
        
    for building in non_collapsed_buildings[:min(len(non_collapsed_buildings), max_buildings_to_plot)]:
        plt.plot(building['centroid_x'], building['centroid_y'], 'bx', markersize=2)
    
    plt.title('Height Difference Map with Building Classification')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='r', label='Collapsed Buildings', 
               markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='x', color='b', label='Non-Collapsed Buildings', 
               markerfacecolor='b', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig('dsm_difference_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved DSM difference map to 'dsm_difference_map.png'")

def visualize_feature_space(features, max_buildings=4000):
    """Visualize the feature space of buildings"""
    print("Creating feature space visualization...")
    # Extract features (limit to max_buildings to avoid memory issues)
    sample_features = features[:min(len(features), max_buildings)]
    
    delta_h = [f['delta_h'] for f in sample_features]
    std_dev = [f['std_dev'] for f in sample_features]
    correlation = [f['correlation'] for f in sample_features]
    status = [f['status'] for f in sample_features]
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'delta_h': delta_h,
        'std_dev': std_dev,
        'correlation': correlation,
        'status': status
    })
    
    # Create plots
    plt.figure(figsize=(18, 16))
    
    # Plot 1: Delta H vs Std Dev
    plt.subplot(2, 2, 1)
    for status, color in zip(['collapsed', 'non_collapsed'], ['red', 'blue']):
        subset = df[df['status'] == status]
        plt.scatter(subset['delta_h'], subset['std_dev'], c=color, label=status, alpha=0.6)
    plt.xlabel('Average Height Difference (m)')
    plt.ylabel('Standard Deviation (m)')
    plt.title('Height Difference vs. Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Delta H vs Correlation
    plt.subplot(2, 2, 2)
    for status, color in zip(['collapsed', 'non_collapsed'], ['red', 'blue']):
        subset = df[df['status'] == status]
        plt.scatter(subset['delta_h'], subset['correlation'], c=color, label=status, alpha=0.6)
    plt.xlabel('Average Height Difference (m)')
    plt.ylabel('Correlation Coefficient')
    plt.title('Height Difference vs. Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Std Dev vs Correlation
    plt.subplot(2, 2, 3)
    for status, color in zip(['collapsed', 'non_collapsed'], ['red', 'blue']):
        subset = df[df['status'] == status]
        plt.scatter(subset['std_dev'], subset['correlation'], c=color, label=status, alpha=0.6)
    plt.xlabel('Standard Deviation (m)')
    plt.ylabel('Correlation Coefficient')
    plt.title('Standard Deviation vs. Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: 3D scatter plot
    ax = plt.subplot(2, 2, 4, projection='3d')
    for status, color in zip(['collapsed', 'non_collapsed'], ['red', 'blue']):
        subset = df[df['status'] == status]
        ax.scatter(subset['delta_h'], subset['std_dev'], subset['correlation'], 
                   c=color, label=status, alpha=0.6)
    ax.set_xlabel('Average Height Difference (m)')
    ax.set_ylabel('Standard Deviation (m)')
    ax.set_zlabel('Correlation Coefficient')
    ax.set_title('3D Feature Space')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('feature_space_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved feature space visualization to 'feature_space_visualization.png'")

def visualize_damage_statistics(collapsed_buildings, non_collapsed_buildings):
    """Visualize damage statistics"""
    print("Creating damage statistics visualization...")
    
    # Calculate total area and percentage
    total_buildings = len(collapsed_buildings) + len(non_collapsed_buildings)
    collapsed_percentage = (len(collapsed_buildings) / total_buildings) * 100
    
    # Calculate total building area
    collapsed_area = sum([b['area'] for b in collapsed_buildings])
    non_collapsed_area = sum([b['area'] for b in non_collapsed_buildings])
    total_area = collapsed_area + non_collapsed_area
    collapsed_area_percentage = (collapsed_area / total_area) * 100
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Building count
    plt.subplot(1, 2, 1)
    counts = [len(non_collapsed_buildings), len(collapsed_buildings)]
    plt.bar(['Non-Collapsed', 'Collapsed'], counts, color=['blue', 'red'])
    plt.title('Building Count')
    plt.ylabel('Number of Buildings')
    
    # Add percentage labels
    for i, count in enumerate(counts):
        percentage = (count / total_buildings) * 100
        plt.text(i, count + 5, f'{count}\n({percentage:.1f}%)', ha='center')
    
    # Plot 2: Building area
    plt.subplot(1, 2, 2)
    areas = [non_collapsed_area, collapsed_area]
    plt.bar(['Non-Collapsed', 'Collapsed'], areas, color=['blue', 'red'])
    plt.title('Building Area')
    plt.ylabel('Total Area (pixels)')
    
    # Add percentage labels
    for i, area in enumerate(areas):
        percentage = (area / total_area) * 100
        plt.text(i, area + 5, f'{area:.0f}\n({percentage:.1f}%)', ha='center')
    
    plt.tight_layout()
    plt.savefig('damage_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved damage statistics to 'damage_statistics.png'")

def train_svm_classifier(features):
    """Train an SVM classifier on the building features - improved with full dataset visualization"""
    print("Training SVM classifier...")
    
    # Prepare data
    X = np.array([[f['delta_h'], f['std_dev'], f['correlation']] for f in features])
    y = np.array([1 if f['status'] == 'collapsed' else 0 for f in features])
    
    # Split data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred_test = svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    
    # Evaluate on full dataset to get complete confusion matrix
    X_all_scaled = scaler.transform(X)
    y_pred_all = svm.predict(X_all_scaled)
    full_accuracy = accuracy_score(y, y_pred_all)
    full_report = classification_report(y, y_pred_all)
    
    print(f"SVM Test Accuracy: {test_accuracy:.4f}")
    print("Test Classification Report:")
    print(test_report)
    
    print(f"SVM Full Dataset Accuracy: {full_accuracy:.4f}")
    print("Full Dataset Classification Report:")
    print(full_report)
    
    # Visualize results
    plt.figure(figsize=(18, 8))
    
    # Test set confusion matrix
    plt.subplot(1, 3, 1)
    cm_test = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Collapsed', 'Collapsed'],
                yticklabels=['Non-Collapsed', 'Collapsed'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix')
    
    # Full dataset confusion matrix
    plt.subplot(1, 3, 2)
    cm_full = confusion_matrix(y, y_pred_all)
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Collapsed', 'Collapsed'],
                yticklabels=['Non-Collapsed', 'Collapsed'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Full Dataset Confusion Matrix')
    
    # Accuracy comparison
    plt.subplot(1, 3, 3)
    plt.bar(['Test Accuracy', 'Full Dataset Accuracy'], [test_accuracy, full_accuracy], color=['green', 'orange'])
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy')
    plt.ylabel('Score')
    
    for i, v in enumerate([test_accuracy, full_accuracy]):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('svm_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved SVM results to 'svm_results.png'")
    
    return svm, scaler, (X_all_scaled, y, y_pred_all)

def main():
    """Main function with memory optimization and essential visualizations"""
    print("Starting collapsed building detection from LiDAR data...")
    start_time = time.time()
    
    # Step 1: Load and process data
    pre_dsm, post_dsm, dsm_params, pre_points, post_points = load_and_process_data(
        PRE_LAZ_PATH, POST_LAZ_PATH, resolution=1.0)
    
    # Free memory
    del pre_points, post_points
    gc.collect()
    
    # Step 2: Extract building footprints (adjusted parameters to get more buildings)
    # Lower min_building_size and height_threshold to detect more buildings
    footprints = simulate_building_footprints(pre_dsm, post_dsm, min_building_size=20, height_threshold=1.8)
    
    # Step 3: Calculate building features
    features = calculate_building_features(pre_dsm, post_dsm, footprints)
    
    # Free memory
    del footprints
    gc.collect()
    
    # Step 4: Detect collapsed buildings
    collapsed_buildings, non_collapsed_buildings = detect_collapsed_buildings(features, threshold=-0.5)
    
    # Free memory
    del features
    gc.collect()
    
    # Step 5: Visualize results
    visualize_dsm_difference(pre_dsm, post_dsm, dsm_params, collapsed_buildings, non_collapsed_buildings)
    
    # Step 6: Create feature space visualization
    all_buildings = collapsed_buildings + non_collapsed_buildings
    visualize_feature_space(all_buildings)
    
    # Step 7: Visualize damage statistics
    visualize_damage_statistics(collapsed_buildings, non_collapsed_buildings)
    
    # Step 8: Train SVM classifier and generate classification report
    if len(collapsed_buildings) > 10 and len(non_collapsed_buildings) > 10:
        train_svm_classifier(all_buildings)
    
    # Print summary statistics
    total_buildings = len(collapsed_buildings) + len(non_collapsed_buildings)
    collapsed_percentage = (len(collapsed_buildings) / total_buildings) * 100
    
    # Calculate total building area
    collapsed_area = sum([b['area'] for b in collapsed_buildings])
    non_collapsed_area = sum([b['area'] for b in non_collapsed_buildings])
    total_area = collapsed_area + non_collapsed_area
    collapsed_area_percentage = (collapsed_area / total_area) * 100
    
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total buildings analyzed: {total_buildings}")
    print(f"Collapsed buildings: {len(collapsed_buildings)} ({collapsed_percentage:.2f}%)")
    print(f"Non-collapsed buildings: {len(non_collapsed_buildings)} ({100-collapsed_percentage:.2f}%)")
    print(f"Percentage of area destroyed: {collapsed_area_percentage:.2f}%")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("Analysis complete!")

if __name__ == "__main__":
    main()