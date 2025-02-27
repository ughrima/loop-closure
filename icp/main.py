import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
import time

# Step 1: Helper Functions
def pad_or_truncate_points(cloud, max_points=300):
    """
    Pad or truncate the point cloud to ensure it has exactly max_points.
    """
    points = np.asarray(cloud.points)
    n_points = len(points)
    if n_points < max_points:
        padding = np.zeros((max_points - n_points, 3))
        points = np.vstack([points, padding])
    elif n_points > max_points:
        indices = np.random.choice(n_points, max_points, replace=False)
        points = points[indices]
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud

def preprocess_point_cloud(cloud, voxel_size=0.1, nb_neighbors=50, std_ratio=1.0, max_points=300):
    """
    Preprocess a point cloud by downsampling, removing outliers, and standardizing the number of points.
    """
    downsampled = cloud.voxel_down_sample(voxel_size)
    cl, _ = downsampled.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    cl = pad_or_truncate_points(cl, max_points)
    return cl

def is_valid_point_cloud(cloud, min_points=10):
    """
    Check if a point cloud is valid (contains at least min_points).
    """
    return len(cloud.points) >= min_points

def compute_fpfh_features(cloud, radius=0.25):
    """
    Compute FPFH features for a point cloud.
    """
    keypoints = cloud.voxel_down_sample(voxel_size=0.1)
    radius_normal = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    cloud.estimate_normals(radius_normal)
    radius_feature = o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=100)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(cloud, radius_feature)
    return fpfh

def coarse_alignment(source, target, fpfh_source, fpfh_target, distance_threshold=0.5):
    """
    Perform coarse alignment using RANSAC and FPFH features.
    """
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, fpfh_source, fpfh_target,
        mutual_filter=False,  # Disable mutual filtering
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999)
    )
    return result.transformation

def icp_alignment(source, target, initial_transformation, threshold=0.02, max_iterations=100):
    """
    Perform Generalized ICP alignment between source and target point clouds.
    """
    result = o3d.pipelines.registration.registration_generalized_icp(
        source, target, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    return result.transformation, result.fitness

def detect_loop_closures_icp(point_clouds, threshold=0.02, fitness_threshold=0.8, max_points=300):
    """
    Detect loop closures using coarse alignment and Generalized ICP.
    """
    n_scans = len(point_clouds)
    loop_closures = []
    fitness_scores = []

    # Compute fitness scores for adaptive thresholding
    for i in range(n_scans):
        for j in range(i + 1, n_scans):
            source = point_clouds[i]
            target = point_clouds[j]

            if not is_valid_point_cloud(source) or not is_valid_point_cloud(target):
                continue

            source = preprocess_point_cloud(source, max_points=max_points)
            target = preprocess_point_cloud(target, max_points=max_points)

            fpfh_source = compute_fpfh_features(source)
            fpfh_target = compute_fpfh_features(target)
            initial_transformation = coarse_alignment(source, target, fpfh_source, fpfh_target)
            transformation, fitness = icp_alignment(source, target, initial_transformation, threshold)

            fitness_scores.append(fitness)

    # Set adaptive threshold based on fitness scores
    adaptive_threshold = np.percentile(fitness_scores, 10)  # Adjust percentile as needed

    # Detect loop closures using adaptive threshold
    for i in tqdm(range(n_scans), desc="Processing scans"):
        for j in range(i + 1, n_scans):
            source = point_clouds[i]
            target = point_clouds[j]

            if not is_valid_point_cloud(source) or not is_valid_point_cloud(target):
                continue

            source = preprocess_point_cloud(source, max_points=max_points)
            target = preprocess_point_cloud(target, max_points=max_points)

            fpfh_source = compute_fpfh_features(source)
            fpfh_target = compute_fpfh_features(target)
            initial_transformation = coarse_alignment(source, target, fpfh_source, fpfh_target)
            transformation, fitness = icp_alignment(source, target, initial_transformation, threshold)

            if fitness > adaptive_threshold:
                loop_closures.append((i, j))

    return loop_closures

def preprocess_folder(input_folder, output_folder, voxel_size=0.1, nb_neighbors=50, std_ratio=1.0, max_points=300):
    """
    Preprocess all .pcd files in a folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each .pcd file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pcd"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Load the point cloud
                cloud = o3d.io.read_point_cloud(input_path)

                # Validate the point cloud
                if not is_valid_point_cloud(cloud):
                    print(f"Skipping invalid point cloud: {input_path}")
                    continue

                # Preprocess the point cloud
                cloud = preprocess_point_cloud(cloud, voxel_size, nb_neighbors, std_ratio, max_points)

                # Save the preprocessed point cloud
                o3d.io.write_point_cloud(output_path, cloud)
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Step 2: Main Execution
if __name__ == "__main__":
    start_time = time.time()

    # Define input folders
    input_folders = [
        os.path.expanduser("~/loop-closure-lpgw/data/instance1_pcd"),
        os.path.expanduser("~/loop-closure-lpgw/data/instance2_pcd"),
        os.path.expanduser("~/loop-closure-lpgw/data/instance3_pcd"),
        os.path.expanduser("~/loop-closure-lpgw/data/instance4_pcd"),
        os.path.expanduser("~/loop-closure-lpgw/data/instance5_pcd"),
    ]

    # Preprocess all folders
    output_folders = [
        os.path.expanduser("~/loop-closure-lpgw/data/preprocessed/icp/icp_instance1_pcd"),
        os.path.expanduser("~/loop-closure-lpgw/data/preprocessed/icp/icp_instance2_pcd"),
        os.path.expanduser("~/loop-closure-lpgw/data/preprocessed/icp/icp_instance3_pcd"),
        os.path.expanduser("~/loop-closure-lpgw/data/preprocessed/icp/icp_instance4_pcd"),
        os.path.expanduser("~/loop-closure-lpgw/data/preprocessed/icp/icp_instance5_pcd"),
    ]

    for input_folder, output_folder in zip(input_folders, output_folders):
        print(f"Preprocessing {input_folder}...")
        preprocess_folder(input_folder, output_folder, max_points=300)
        print(f"Saved preprocessed data to {output_folder}")

    # Load preprocessed point clouds
    all_point_clouds = []
    total_points = 0

    for folder in output_folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist. Skipping...")
            continue

        pcd_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pcd")]
        for file in pcd_files:
            try:
                cloud = o3d.io.read_point_cloud(file)
                if is_valid_point_cloud(cloud):
                    all_point_clouds.append(cloud)
                    total_points += len(cloud.points)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    print(f"Loaded {len(all_point_clouds)} point clouds from {len(output_folders)} folders.")
    print(f"Total points across all point clouds: {total_points}")

    # Process in batches to avoid memory overload
    batch_size = 50
    loop_closures = []

    for i in range(0, len(all_point_clouds), batch_size):
        batch_point_clouds = all_point_clouds[i:i + batch_size]
        batch_start_time = time.time()
        batch_loop_closures = detect_loop_closures_icp(batch_point_clouds, fitness_threshold=0.7, max_points=300)
        batch_time = time.time() - batch_start_time
        loop_closures.extend([(i + idx1, i + idx2) for idx1, idx2 in batch_loop_closures])
        print(f"Processed batch {i // batch_size + 1} with {len(batch_point_clouds)} point clouds in {batch_time:.2f} seconds.")

    # Save results
    np.save("icp_detected_loop_closures.npy", loop_closures)
    total_time = time.time() - start_time

    print(f"Detected {len(loop_closures)} loop closures.")
    print(f"Total time taken: {total_time:.2f} seconds.")