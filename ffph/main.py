# import open3d as o3d
# import numpy as np
# import os
# from tqdm import tqdm
# import time

# # Step 1: Preprocessing
# def preprocess_point_cloud(cloud, max_points=300, voxel_size=0.1, nb_neighbors=50, std_ratio=1.0):
#     """
#     Preprocess a point cloud by downsampling, removing outliers, and standardizing the number of points.
#     """
#     # Downsample the point cloud
#     downsampled = cloud.voxel_down_sample(voxel_size)
    
#     # Remove outliers
#     cl, _ = downsampled.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
#     # Standardize the number of points
#     points = np.asarray(cl.points)
#     if len(points) < max_points:
#         padding = np.zeros((max_points - len(points), 3))
#         points = np.vstack([points, padding])
#     elif len(points) > max_points:
#         indices = np.random.choice(len(points), max_points, replace=False)
#         points = points[indices]
#     cl.points = o3d.utility.Vector3dVector(points)
#     return cl

# def is_valid_point_cloud(cloud, min_points=10):
#     """
#     Check if a point cloud is valid (contains at least min_points).
#     """
#     return len(cloud.points) >= min_points

# # Step 2: Feature Extraction (FPFH)
# def compute_fpfh_features(cloud, radius=0.25):
#     """
#     Compute FPFH features for a point cloud.
#     """
#     # Estimate normals
#     cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
#     # Compute FPFH features
#     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         cloud,
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
#     )
#     # Convert memoryview to NumPy array
#     return np.asarray(fpfh.data).T  # Transpose to make it easier to work with

# # Step 3: Feature Matching
# def match_fpfh_features(fpfh1, fpfh2, threshold=0.98):
#     """
#     Match FPFH features between two point clouds using cosine similarity.
#     """
#     similarity = np.dot(fpfh1, fpfh2.T)  # Compute cosine similarity
#     epsilon = 1e-8
#     norm1 = np.linalg.norm(fpfh1, axis=1)[:, None] + epsilon
#     norm2 = np.linalg.norm(fpfh2, axis=1)[None, :] + epsilon
#     similarity /= norm1
#     similarity /= norm2
#     matches = np.where(similarity > threshold)
#     return matches

# # Step 4: Loop Closure Detection
# def detect_loop_closures_fpfh(point_clouds, radius=0.25, threshold=0.98, window_size=10):
#     """
#     Detect loop closures using FPFH descriptors.
#     """
#     n_scans = len(point_clouds)
#     loop_closures = []
    
#     # Compute FPFH features sequentially
#     print("Computing FPFH features...")
#     fpfh_features = []
#     for cloud in tqdm(point_clouds, desc="Processing point clouds"):
#         fpfh = compute_fpfh_features(cloud, radius)
#         fpfh_features.append(fpfh)  # Already a NumPy array
    
#     # Compare FPFH features pairwise within a sliding window
#     print("Detecting loop closures...")
#     for i in tqdm(range(n_scans), desc="Detecting loop closures"):
#         for j in range(i + 1, min(i + window_size, n_scans)):
#             matches = match_fpfh_features(fpfh_features[i], fpfh_features[j], threshold)
#             if len(matches[0]) > 0:  # If there are matches
#                 loop_closures.append((i, j))
    
#     return loop_closures

# # Main Execution
# if __name__ == "__main__":
#     start_time = time.time()
    
#     # Define input folders
#     input_folders = [
#         os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance1_pcd"),
#         os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance2_pcd"),
#         os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance3_pcd"),
#         os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance4_pcd"),
#         os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance5_pcd"),
#     ]
    
#     # Load preprocessed point clouds
#     all_point_clouds = []
#     total_points = 0
#     for folder in input_folders:
#         if not os.path.exists(folder):
#             print(f"Warning: Folder {folder} does not exist. Skipping...")
#             continue
        
#         pcd_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pcd")]
#         for file in pcd_files:
#             try:
#                 cloud = o3d.io.read_point_cloud(file)
#                 if is_valid_point_cloud(cloud):
#                     cloud = preprocess_point_cloud(cloud)
#                     all_point_clouds.append(cloud)
#                     total_points += len(cloud.points)
#                 else:
#                     print(f"Skipping invalid point cloud: {file}")
#             except Exception as e:
#                 print(f"Error loading {file}: {e}")
    
#     print(f"Loaded {len(all_point_clouds)} point clouds from {len(input_folders)} folders.")
#     print(f"Total points across all point clouds: {total_points}")
    
#     # Detect loop closures using FPFH
#     loop_closures = detect_loop_closures_fpfh(all_point_clouds)
    
#     # Save results
#     np.save("fpfh_detected_loop_closures.npy", loop_closures)
#     total_time = time.time() - start_time
    
#     print(f"Detected {len(loop_closures)} loop closures.")
#     print(f"Total time taken: {total_time:.2f} seconds.")


import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
import time
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

# Step 1: Preprocessing
def preprocess_point_cloud(cloud, max_points=300, voxel_size=0.1, nb_neighbors=50, std_ratio=1.0):
    """
    Preprocess a point cloud by downsampling, removing outliers, and standardizing the number of points.
    """
    # Downsample the point cloud
    downsampled = cloud.voxel_down_sample(voxel_size)
    
    # Remove outliers
    cl, _ = downsampled.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    # Standardize the number of points
    points = np.asarray(cl.points)
    if len(points) < max_points:
        padding = np.zeros((max_points - len(points), 3))
        points = np.vstack([points, padding])
    elif len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    cl.points = o3d.utility.Vector3dVector(points)
    return cl

def is_valid_point_cloud(cloud, min_points=10):
    """
    Check if a point cloud is valid (contains at least min_points).
    """
    return len(cloud.points) >= min_points

# Step 2: Feature Extraction (FPFH)
def compute_fpfh_features(cloud, radius=0.25):
    """
    Compute FPFH features for a point cloud.
    """
    # Estimate normals
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )
    # Convert memoryview to NumPy array
    return np.asarray(fpfh.data).T  # Transpose to make it easier to work with

# Step 3: Dimensionality Reduction (PCA)
def reduce_dimensionality(features, variance_threshold=0.95):
    """
    Reduce dimensionality of FPFH features using PCA.
    """
    pca = PCA(n_components=None)
    pca.fit(features)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    reduced_features = PCA(n_components=n_components).fit_transform(features)
    print(f"Reduced FPFH features to {n_components} dimensions.")
    return reduced_features

# Step 4: Loop Closure Detection Using KD-Trees
def detect_loop_closures_fpfh(point_clouds, radius=0.25, k_neighbors=10, percentile=95):
    """
    Detect loop closures using FPFH descriptors with PCA, adaptive thresholding, and KD-Trees.
    """
    n_scans = len(point_clouds)
    loop_closures = []
    
    # Compute FPFH features sequentially
    print("Computing FPFH features...")
    fpfh_features = []
    for cloud in tqdm(point_clouds, desc="Processing point clouds"):
        fpfh = compute_fpfh_features(cloud, radius)
        fpfh_features.append(fpfh)  # Already a NumPy array
    
    # Flatten and reduce dimensionality of FPFH features
    print("Flattening and reducing dimensionality of FPFH features...")
    flattened_features = np.vstack(fpfh_features)
    reduced_features = reduce_dimensionality(flattened_features, variance_threshold=0.95)
    
    # Save reduced features for debugging
    np.save("fpfh_reduced_features.npy", reduced_features)
    print("Saved reduced FPFH features to fpfh_reduced_features.npy")
    
    # Build KD-Tree for fast nearest neighbor search
    print("Building KD-Tree for fast nearest neighbor search...")
    tree = KDTree(reduced_features)
    
    # Detect loop closures using KD-Tree
    print("Detecting loop closures...")
    distances, indices = tree.query(reduced_features, k=k_neighbors)
    print(f"Distances shape: {distances.shape}")
    print(f"Indices shape: {indices.shape}")
    
    threshold = np.percentile(distances[:, 1:], percentile)  # Ignore self-match at index 0
    print(f"Adaptive threshold set to {threshold:.4f}")
    
    for i in tqdm(range(n_scans), desc="Processing scans"):
        for idx, j in enumerate(indices[i][1:]):  # Skip self-match at index 0
            if distances[i, idx + 1] < threshold:  # Directly access the distance
                loop_closures.append((i, j))
    
    return loop_closures

# Main Execution
if __name__ == "__main__":
    start_time = time.time()
    
    # Define input folders
    input_folders = [
        os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance1_pcd"),
        os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance2_pcd"),
        os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance3_pcd"),
        os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance4_pcd"),
        os.path.expanduser("/home/agrima/loop-closure-lpgw/data/preprocessed/icp/icp_instance5_pcd"),
    ]
    
    # Load preprocessed point clouds
    all_point_clouds = []
    total_points = 0
    for folder in input_folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist. Skipping...")
            continue
        
        pcd_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pcd")]
        for file in pcd_files:
            try:
                cloud = o3d.io.read_point_cloud(file)
                if is_valid_point_cloud(cloud):
                    cloud = preprocess_point_cloud(cloud)
                    all_point_clouds.append(cloud)
                    total_points += len(cloud.points)
                else:
                    print(f"Skipping invalid point cloud: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    print(f"Loaded {len(all_point_clouds)} point clouds from {len(input_folders)} folders.")
    print(f"Total points across all point clouds: {total_points}")
    
    # Process in batches to avoid memory overload
    batch_size = 50
    loop_closures = []

    for i in range(0, len(all_point_clouds), batch_size):
        batch_point_clouds = all_point_clouds[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} with {len(batch_point_clouds)} point clouds...")
        batch_loop_closures = detect_loop_closures_fpfh(batch_point_clouds)
        loop_closures.extend([(i + idx1, i + idx2) for idx1, idx2 in batch_loop_closures])
    
    # Save results
    np.save("fpfh_detected_loop_closures.npy", loop_closures)
    total_time = time.time() - start_time
    
    print(f"Detected {len(loop_closures)} loop closures.")
    print(f"Total time taken: {total_time:.2f} seconds.")