import numpy as np
from ot.gromov import gromov_wasserstein
from joblib import Parallel, delayed
import time
from tqdm import tqdm  # Import tqdm for the progress bar

# Load reference space
reference_distance_matrix = np.load("/home/agrima/lidar_slam_project/gw_reference_distance_matrix.npy")
reference_prob_dist = np.load("/home/agrima/lidar_slam_project/gw_reference_prob_dist.npy")

# Load all distance matrices and probability distributions
all_distance_matrices = []
all_probability_distributions = []
for i in range(0, 2500, 100):
    distance_matrices = np.load(f"distance_matrices_{i}.npy")
    probability_distributions = np.load(f"probability_distributions_{i}.npy")
    all_distance_matrices.extend(distance_matrices)
    all_probability_distributions.extend(probability_distributions)

# Function to compute Gromov-Wasserstein distance
def compute_gw(distance_matrix, prob_dist):
    return gromov_wasserstein(
        distance_matrix, reference_distance_matrix, prob_dist, reference_prob_dist
    )

# Compute GW embeddings in parallel with a progress bar
start_time = time.time()
print("Starting GW computation...")

gw_embeddings = Parallel(n_jobs=2)(  # Reduced from 4 to 2 to save memory
    delayed(compute_gw)(distance_matrix, prob_dist)
    for distance_matrix, prob_dist in tqdm(
        zip(all_distance_matrices, all_probability_distributions),
        total=len(all_distance_matrices),
        desc="Processing",
    )
)

# Print total time taken
total_time = time.time() - start_time
print(f"Computation completed in {total_time / 60:.2f} minutes.")

# Save GW embeddings
np.save("gw_embeddings.npy", np.array(gw_embeddings))
print("Saved GW embeddings.")