import open3d as o3d
import numpy as np
import os

def downsample_point_cloud(cloud, voxel_size=0.1):
    """
    Downsample the point cloud using a voxel grid filter.
    """
    return cloud.voxel_down_sample(voxel_size)

def remove_outliers(cloud, nb_neighbors=50, std_ratio=1.0):
    """
    Remove outliers using a statistical outlier removal filter.
    """
    cl, _ = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl

def preprocess_scan(input_path, output_path, voxel_size=0.1, nb_neighbors=50, std_ratio=1.0):
    """
    Preprocess a single LiDAR scan:
    1. Load the .pcd file.
    2. Downsample the point cloud.
    3. Remove outliers.
    4. Save the preprocessed point cloud.
    """
    # Load the point cloud
    cloud = o3d.io.read_point_cloud(input_path)

    # Downsample the point cloud
    cloud = downsample_point_cloud(cloud, voxel_size)

    # Remove outliers
    cloud = remove_outliers(cloud, nb_neighbors, std_ratio)

    # Save the preprocessed point cloud
    o3d.io.write_point_cloud(output_path, cloud)

def preprocess_folder(input_folder, output_folder, voxel_size=0.1, nb_neighbors=50, std_ratio=1.0):
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
            preprocess_scan(input_path, output_path, voxel_size, nb_neighbors, std_ratio)
            print("Processed {}".format(filename))

if __name__ == "__main__":
    # Define input and output folders
    input_folders = [
        os.path.expanduser("~/lidar_slam_project/data/instance1_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/instance2_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/instance3_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/instance4_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/instance5_pcd"),
    ]
    output_folders = [
        os.path.expanduser("~/lidar_slam_project/data/preprocessed_gw/instance1_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/preprocessed_gw/instance2_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/preprocessed_gw/instance3_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/preprocessed_gw/instance4_pcd"),
        os.path.expanduser("~/lidar_slam_project/data/preprocessed_gw/instance5_pcd"),
    ]

    # Preprocess each folder
    for input_folder, output_folder in zip(input_folders, output_folders):
        print("Processing {}...".format(input_folder))
        preprocess_folder(input_folder, output_folder)
        print("Saved preprocessed data to {}".format(output_folder))
