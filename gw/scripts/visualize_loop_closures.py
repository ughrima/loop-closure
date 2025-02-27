import open3d as o3d
import numpy as np
import os
import re
import random

def load_scan(file_path):
    """
    Load a preprocessed .pcd file.
    """
    return o3d.io.read_point_cloud(file_path)

def visualize_and_save_loop_closure(scan_i, scan_j, pcd_files, output_folder):
    """
    Visualize a pair of scans identified as a loop closure and save the visualization as a PNG image.
    """
    # Load the two scans
    cloud_i = load_scan(pcd_files[scan_i])
    cloud_j = load_scan(pcd_files[scan_j])
    
    # Assign colors to differentiate the scans
    cloud_i.paint_uniform_color([1, 0, 0])  # Red for scan i
    cloud_j.paint_uniform_color([0, 0, 1])  # Blue for scan j
    
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the point clouds to the visualizer
    vis.add_geometry(cloud_i)
    vis.add_geometry(cloud_j)
    
    # Run the visualizer to render the scene
    vis.run()
    
    # Capture the screen image and save it as a PNG file
    output_path = os.path.join(output_folder, f"gw_loop_closure_{scan_i}_{scan_j}.png")
    vis.capture_screen_image(output_path)
    print(f"Saved GW visualization to {output_path}")
    
    # Close the visualizer
    vis.destroy_window()

def get_sorted_pcd_files(folder_path):
    """
    Get all .pcd files from the folder and sort them numerically.
    """
    # Get all files in the folder
    files = os.listdir(folder_path)
    
    # Filter for .pcd files
    pcd_files = [f for f in files if f.endswith(".pcd")]
    
    # Sort files numerically (e.g., scan_0001.pcd, scan_0002.pcd, etc.)
    pcd_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    
    # Add the full path to each file
    pcd_files = [os.path.join(folder_path, f) for f in pcd_files]
    
    return pcd_files

if __name__ == "__main__":
    # Path to the folder containing .pcd files
    folder_path = "data/preprocessed/instance1_pcd"  # Update this path as needed
    
    # Path to the output folder for saving PNG images
    output_folder = "gw_visualizations"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created GW output folder at: {output_folder}")
    else:
        print(f"GW output folder already exists at: {output_folder}")
    
    # Get all .pcd files from the folder
    pcd_files = get_sorted_pcd_files(folder_path)
    num_pcd_files = len(pcd_files)
    print(f"Found {num_pcd_files} .pcd files in {folder_path}.")
    
    # Load filtered loop closures
    try:
        filtered_closures = np.load("/home/agrima/loop-closure-lpgw/gw/gw_filtered_detected_loop_closures.npy", allow_pickle=True)
        print(f"Loaded {len(filtered_closures)} GW-based filtered loop closures.")
    except FileNotFoundError:
        print("Error: gw_filtered_detected_loop_closures.npy not found.")
        exit(1)
    except Exception as e:
        print("Error loading GW-based filtered loop closures:", e)
        exit(1)
    
    # Filter closures to ensure indices are valid
    valid_closures = [
        (scan_i, scan_j) for scan_i, scan_j in filtered_closures
        if scan_i < num_pcd_files and scan_j < num_pcd_files
    ]
    print(f"Valid GW-based loop closures for visualization: {len(valid_closures)}")
    
    # Randomly sample 10 loop closures for visualization
    num_samples = 10
    sampled_closures = random.sample(valid_closures, min(num_samples, len(valid_closures)))
    print(f"Randomly selected {len(sampled_closures)} GW-based loop closures for visualization.")
    
    # Visualize and save the sampled closures
    for i, (scan_i, scan_j) in enumerate(sampled_closures):
        print(f"Visualizing and saving GW-based loop closure {i+1}: Scan {scan_i} and Scan {scan_j}")
        visualize_and_save_loop_closure(scan_i, scan_j, pcd_files, output_folder)