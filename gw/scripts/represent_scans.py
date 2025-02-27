import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
import os

def load_scan(file_path):
    """
    Load a preprocessed .pcd file.
    """
    return o3d.io.read_point_cloud(file_path)

def pad_or_truncate_points(points, max_points=300):  # Standardized with LPGW
    """
    Pad or truncate the point cloud to ensure it has exactly max_points.
    """
    n_points = len(points)
    if n_points < max_points:
        # Pad with zeros
        padding = np.zeros((max_points - n_points, 3))
        points = np.vstack([points, padding])
    elif n_points > max_points:
        # Truncate
        indices = np.random.choice(n_points, max_points, replace=False)
        points = points[indices]
    return points

def compute_distance_matrix(points):
    """
    Compute the pairwise Euclidean distance matrix for a set of points.
    """
    return cdist(points, points, metric='euclidean')

def compute_probability_distribution(points):
    """
    Compute a uniform probability distribution over the points.
    """
    n_points = len(points)
    return np.ones(n_points) / n_points

def process_scan(file_path, max_points=300):  # Standardized with LPGW
    """
    Process a single scan:
    1. Load the scan.
    2. Extract points.
    3. Pad or truncate to max_points.
    4. Compute distance matrix.
    5. Compute probability distribution.
    """
    # Load the scan
    cloud = load_scan(file_path)
    points = np.asarray(cloud.points)

    # Pad or truncate the points
    points = pad_or_truncate_points(points, max_points)

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(points)

    # Compute probability distribution
    prob_dist = compute_probability_distribution(points)

    return distance_matrix, prob_dist

def process_folder(input_folder, max_points=300, batch_size=50):  # Standardized batch size
    """
    Process all .pcd files in a folder in batches.
    """
    distance_matrices = []
    probability_distributions = []

    # Get list of .pcd files
    pcd_files = [f for f in os.listdir(input_folder) if f.endswith(".pcd")]

    # Process in batches
    for i in range(0, len(pcd_files), batch_size):
        batch_files = pcd_files[i:i + batch_size]
        for filename in batch_files:
            file_path = os.path.join(input_folder, filename)
            distance_matrix, prob_dist = process_scan(file_path, max_points)
            distance_matrices.append(distance_matrix)
            probability_distributions.append(prob_dist)
            print(f"Processed {filename}")

    return distance_matrices, probability_distributions

if __name__ == "__main__":
    # Define input folders
    input_folders = [
        os.path.expanduser("~/lidar_slam_project/data/preprocessed/instance1_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/preprocessed/instance2_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/preprocessed/instance3_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/preprocessed/instance4_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/preprocessed/instance5_pcd"),
    ]

    # Process each folder
    all_distance_matrices = []
    all_probability_distributions = []

    for input_folder in input_folders:
        print(f"Processing {input_folder} for GW...")
        distance_matrices, probability_distributions = process_folder(input_folder)
        all_distance_matrices.extend(distance_matrices)
        all_probability_distributions.extend(probability_distributions)

    # Save results in smaller batches
    for i in range(0, len(all_distance_matrices), 100):
        batch_distance_matrices = all_distance_matrices[i:i + 100]
        batch_probability_distributions = all_probability_distributions[i:i + 100]
        np.save(f"distance_matrices_{i}.npy", np.array(batch_distance_matrices))  # Standardized file names
        np.save(f"probability_distributions_{i}.npy", np.array(batch_probability_distributions))  # Standardized file names
        print(f"Saved batch {i} to {i + 100}")

    print("Saved all distance matrices and probability distributions for GW.")
