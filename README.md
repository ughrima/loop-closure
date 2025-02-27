
# Loop Closure Detection Results

## Overview
This document summarizes the results of loop closure detection using three different methods: **Fast Point Feature Histograms (FPFH)**, **Gromov-Wasserstein (GW)**, and **Iterative Closest Point (ICP)**. The dataset consists of five `.bag` files containing point cloud data from a RealSense sensor. Each method was applied to detect loop closures, and the results were evaluated based on the number of detected loop closures, their characteristics, and performance trade-offs.

---

### **1. Dataset Description**

The dataset consists of five `.bag` files, each containing point cloud data captured by a RealSense sensor. Below are the key statistics for each file:

| File Index | Duration (sec) | Size (GB) | Messages |
|------------|----------------|-----------|----------|
| 1          | 19.9           | 1.3       | 391      |
| 2          | 19.8           | 1.5       | 441      |
| 3          | 27.0           | 2.0       | 557      |
| 4          | 27.3           | 2.1       | 577      |
| 5          | 26.0           | 1.0       | 544      |

- **Duration**: The duration of the recorded data ranges from approximately 19.8 to 27.3 seconds.
- **Size**: The size of the `.bag` files varies between 1.0 GB and 2.1 GB.
- **Messages**: The number of messages in each file ranges from 391 to 577, indicating varying levels of data density.


---

### **2. Methodology**

#### **FPFH Pipeline**
- **Preprocessing**: Point clouds are downsampled, outliers are removed, and the number of points is standardized.
- **Feature Extraction**: FPFH features are computed for each point cloud.
- **Dimensionality Reduction**: PCA is used to reduce the dimensionality of the FPFH features.
- **Loop Closure Detection**: A KD-Tree is built for fast nearest neighbor search, and an adaptive threshold is set based on the distances between features.

#### **GW Pipeline**
- **Preprocessing**: Point clouds are downsampled and outliers are removed.
- **Distance Matrix Computation**: Pairwise Euclidean distance matrices are computed for each point cloud.
- **Probability Distribution**: A uniform probability distribution is computed over the points.
- **Gromov-Wasserstein Distance**: The Gromov-Wasserstein distance is computed between each point cloud and a reference point cloud.
- **Dimensionality Reduction**: PCA is used to reduce the dimensionality of the embeddings.
- **Loop Closure Detection**: A KD-Tree is used to find the nearest neighbors and detect loop closures.

#### **ICP Pipeline**
- **Preprocessing**: Point clouds are downsampled and outliers are removed.
- **Coarse Alignment**: Coarse alignment is performed using RANSAC and FPFH features.
- **ICP Alignment**: Generalized ICP is used for fine alignment.
- **Loop Closure Detection**: Loop closures are detected based on the fitness score of the alignment.

---

### **3. Results**

#### **FPFH Results**
- **Number of Loop Closures Detected**: **19,988**
- **Example Pairs**:
    ```python
    [[0, 180], [0, 5546], [0, 5621], [0, 11542], [0, 3766],
     [0, 11458], [0, 2577], [0, 11414], [0, 2412], [1, 2563],
     [1, 3794], [1, 11734], [1, 10534], [1, 6397], [1, 10657],
     [1, 9909], [1, 8512], [2, 13318], [2, 13829], [2, 10339]]
    ```
- **Inference**: 
  - FPFH detects a large number of loop closures, many involving point cloud 0.
  - This suggests that FPFH has **high recall** but may include many **false positives**, especially for specific point clouds like index 0.

#### **GW Results**
- **Number of Loop Closures Detected**: **1,103**
- **Example Pairs**:
    ```python
    [[4, 386], [4, 68], [4, 823], [4, 1002], [4, 62],
     [4, 2316], [4, 1409], [4, 1097], [4, 62], [4, 1002],
     [4, 836], [4, 386], [4, 1002], [4, 1409], [4, 386],
     [4, 1002], [4, 836], [4, 386], [4, 1002], [4, 836]]
    ```
- **Inference**: 
  - GW detects significantly fewer loop closures compared to FPFH and ICP.
  - Many involve point cloud 4, indicating that GW might have **higher precision** but **lower recall**, as it is more conservative in identifying loop closures.

#### **ICP Results**
- **Number of Loop Closures Detected**: **46,111**
- **Example Pairs**:
    ```python
    [[0, 2], [0, 3], [0, 4], [0, 5], [0, 7],
     [0, 8], [0, 13], [0, 15], [0, 16], [0, 17],
     [0, 20], [0, 22], [0, 23], [0, 24], [0, 25],
     [0, 26], [0, 27], [0, 28], [0, 29], [0, 30]]
    ```
- **Inference**: 
  - ICP detects the most loop closures, many involving point cloud 0.
  - This suggests that ICP has **high recall** but may include many **false positives**, indicating **lower precision**.

---

### **4. Observations and Recommendations**

#### **FPFH**
- **Strengths**: High recall, capable of detecting a large number of potential loop closures.
- **Weaknesses**: Potential for many false positives, especially involving point cloud 0.
- **Recommendation**: Consider adjusting the adaptive threshold or using additional filtering techniques to reduce false positives.

#### **GW**
- **Strengths**: Higher precision, more conservative in detecting loop closures.
- **Weaknesses**: Lower recall, may miss some valid loop closures.
- **Recommendation**: Adjust the variance threshold in PCA to balance precision and recall.

#### **ICP**
- **Strengths**: High recall, capable of detecting a large number of potential loop closures.
- **Weaknesses**: Potential for many false positives, especially involving point cloud 0.
- **Recommendation**: Evaluate the fitness threshold used in ICP to improve precision.

---

### **5. Conclusion**
The three methods exhibit different behaviors in detecting loop closures:
- **FPFH**: High recall, potential for false positives.
- **GW**: Higher precision, lower recall.
- **ICP**: High recall, potential for false positives.

