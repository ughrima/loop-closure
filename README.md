# Loop Closure Detection 

This repository contains implementations of three pipelines for loop closure detection in point cloud data: **FPFH Pipeline**, **GW Pipeline**, and **ICP Pipeline**. Below is a detailed description of each pipeline.

---

## FPFH Pipeline

### Overview
The FPFH (Fast Point Feature Histogram) pipeline is designed to detect loop closures by extracting and comparing geometric features from point clouds.

### Steps
1. **Preprocessing**
   - Point clouds are downsampled.
   - Outliers are removed.
   - The number of points is standardized.

2. **Feature Extraction**
   - FPFH features are computed for each point cloud.

3. **Dimensionality Reduction**
   - PCA (Principal Component Analysis) is used to reduce the dimensionality of the FPFH features.

4. **Loop Closure Detection**
   - A KD-Tree is built for fast nearest neighbor search.
   - An adaptive threshold is set based on the distances between features to detect loop closures.

---

## GW Pipeline

### Overview
The GW (Gromov-Wasserstein) pipeline computes embeddings for point clouds using the Gromov-Wasserstein distance and detects loop closures based on these embeddings.

### Steps
1. **Preprocessing**
   - Point clouds are downsampled.
   - Outliers are removed.

2. **Distance Matrix Computation**
   - Pairwise Euclidean distance matrices are computed for each point cloud.

3. **Probability Distribution**
   - A uniform probability distribution is computed over the points.

4. **Gromov-Wasserstein Distance**
   - The Gromov-Wasserstein distance is computed between each point cloud and a reference point cloud.

5. **Dimensionality Reduction**
   - PCA is used to reduce the dimensionality of the embeddings.

6. **Loop Closure Detection**
   - A KD-Tree is used to find the nearest neighbors and detect loop closures.

---

## ICP Pipeline

### Overview
The ICP (Iterative Closest Point) pipeline aligns point clouds using coarse and fine alignment techniques and detects loop closures based on the fitness score of the alignment.

### Steps
1. **Preprocessing**
   - Point clouds are downsampled.
   - Outliers are removed.

2. **Coarse Alignment**
   - Coarse alignment is performed using RANSAC and FPFH features.

3. **ICP Alignment**
   - Generalized ICP is used for fine alignment.

4. **Loop Closure Detection**
   - Loop closures are detected based on the fitness score of the alignment.
