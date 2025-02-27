# import numpy as np
# from scipy.spatial.distance import cdist
# from scipy.spatial import KDTree
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from joblib import Parallel, delayed
# from tqdm import tqdm
# import time

# # Load GW embeddings
# try:
#     gw_embeddings = np.load("gw_embeddings.npy")
#     print("Loaded GW embeddings with shape:", gw_embeddings.shape)
# except FileNotFoundError:
#     print("Error: gw_embeddings.npy not found.")
#     exit(1)
# except Exception as e:
#     print("Error loading GW embeddings:", e)
#     exit(1)

# # Flatten the embeddings into 1D vectors
# def flatten_embeddings(embeddings):
#     return embeddings.reshape(embeddings.shape[0], -1)

# # Reduce dimensionality dynamically using PCA
# def reduce_dimensionality(embeddings, variance_threshold=0.90):  # Increased variance threshold
#     scaler = StandardScaler()
#     embeddings_scaled = scaler.fit_transform(embeddings)  # Standardization
#     pca = PCA(n_components=None)  # Auto-detect based on variance
#     pca.fit(embeddings_scaled)
#     cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
#     n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
#     print(f"Selected {n_components} PCA components to retain {variance_threshold * 100}% variance.")
#     pca = PCA(n_components=n_components)
#     reduced_embeddings = pca.fit_transform(embeddings_scaled)
#     print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
#     print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")
#     return reduced_embeddings

# # Detect loop closures using KD-Tree for efficiency
# def detect_loop_closure(embeddings, k_neighbors=10, percentile=10):  # Increased percentile
#     n_scans = len(embeddings)
#     loop_closures = []
#     print("Building KD-Tree for fast nearest neighbor search...")
#     start_time = time.time()
#     tree = KDTree(embeddings)
#     print(f"KD-Tree built in {time.time() - start_time:.2f} seconds.")
#     print("Finding nearest neighbors...")
#     distances, indices = tree.query(embeddings, k=k_neighbors)
#     threshold = np.percentile(distances[:, 1:], percentile)  # Ignore self-match at index 0
#     print(f"Adaptive threshold set to {threshold:.4f}")
#     print("Detecting loop closures...")
#     for i in tqdm(range(n_scans), desc="Processing scans"):
#         for j in indices[i][1:]:  # Skip self-match at index 0
#             if distances[i, np.where(indices[i] == j)][0] < threshold:
#                 loop_closures.append((i, j))
#     print(f"Loop closure detection completed. Detected: {len(loop_closures)}")
#     return loop_closures

# # Flatten the embeddings
# flattened_embeddings = flatten_embeddings(gw_embeddings)
# print("Flattened embeddings shape:", flattened_embeddings.shape)

# # Reduce dimensionality dynamically
# reduced_embeddings = reduce_dimensionality(flattened_embeddings, variance_threshold=0.95)

# # Normalize embeddings for distance computation
# scaler = StandardScaler()
# normalized_embeddings = scaler.fit_transform(reduced_embeddings)

# # Detect loop closures using KD-Tree
# print("Starting loop closure detection...")
# start_time = time.time()
# loop_closures = detect_loop_closure(normalized_embeddings, k_neighbors=10, percentile=10)

# # Save detected loop closures
# np.save("gw_filtered_detected_loop_closures.npy", loop_closures)
# print(f"Saved {len(loop_closures)} detected loop closures to gw_filtered_detected_loop_closures.npy.")
# print(f"Total time taken: {time.time() - start_time:.2f} seconds.")

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

# Load GW embeddings
try:
    gw_embeddings = np.load("/home/agrima/loop-closure-lpgw/gw/gw_embeddings.npy")
    print("Loaded GW embeddings with shape:", gw_embeddings.shape)
except FileNotFoundError:
    print("Error: gw_embeddings.npy not found.")
    exit(1)
except Exception as e:
    print("Error loading GW embeddings:", e)
    exit(1)

# Flatten the embeddings into 1D vectors
def flatten_embeddings(embeddings):
    return embeddings.reshape(embeddings.shape[0], -1)

# Reduce dimensionality dynamically using PCA
def reduce_dimensionality(embeddings, variance_threshold=0.90):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)  # Standardization

    pca = PCA(n_components=None)  # Auto-detect based on variance
    pca.fit(embeddings_scaled)

    # Determine number of components to retain the required variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    print(f"Selected {n_components} PCA components to retain {variance_threshold * 100}% variance.")

    # Apply PCA with the selected components
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_scaled)

    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")
    return reduced_embeddings

# Detect loop closures using KD-Tree for efficiency
def detect_loop_closure(embeddings, k_neighbors=10, percentile=5):
    n_scans = len(embeddings)
    loop_closures = []
    
    print("Building KD-Tree for fast nearest neighbor search...")
    start_time = time.time()
    tree = KDTree(embeddings)
    print(f"KD-Tree built in {time.time() - start_time:.2f} seconds.")

    print("Finding nearest neighbors...")
    distances, indices = tree.query(embeddings, k=k_neighbors)

    # Adaptive thresholding
    threshold = np.percentile(distances[:, 1:], percentile)  # Ignore self-match at index 0
    print(f"Adaptive threshold set to {threshold:.4f}")

    print("Detecting loop closures...")
    for i in tqdm(range(n_scans), desc="Processing scans"):
        for j in indices[i][1:]:  # Skip self-match at index 0
            if distances[i, np.where(indices[i] == j)][0] < threshold:
                loop_closures.append((i, j))

    print(f"Loop closure detection completed. Detected: {len(loop_closures)}")
    return loop_closures

# Flatten the embeddings
flattened_embeddings = flatten_embeddings(gw_embeddings)
print("Flattened embeddings shape:", flattened_embeddings.shape)

# Reduce dimensionality dynamically
reduced_embeddings = reduce_dimensionality(flattened_embeddings, variance_threshold=0.90)

# Normalize embeddings for distance computation
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(reduced_embeddings)

# Detect loop closures using KD-Tree
print("Starting loop closure detection...")
start_time = time.time()
loop_closures = detect_loop_closure(normalized_embeddings, k_neighbors=10, percentile=5)

# Save detected loop closures
np.save("gw_filtered_detected_loop_closures.npy", loop_closures)
print(f"Saved {len(loop_closures)} detected loop closures to gw_filtered_detected_loop_closures.npy.")
print(f"Total time taken: {time.time() - start_time:.2f} seconds.")
