import numpy as np

# Load the first scan as the reference
reference_distance_matrix = np.load("/home/agrima/lidar_slam_project/distance_matrices_0.npy")[0]
reference_prob_dist = np.load("/home/agrima/lidar_slam_project/probability_distributions_0.npy")[0]

# Save the reference space
np.save("gw_reference_distance_matrix.npy", reference_distance_matrix)
np.save("gw_reference_prob_dist.npy", reference_prob_dist)

print("Saved GW reference space.")