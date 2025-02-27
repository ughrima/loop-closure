import matplotlib.pyplot as plt
import numpy as np

# Load original and filtered closures
original_closures = np.load("/home/agrima/lidar_slam_project/lpgw/filtered_detected_loop_closures.npy", allow_pickle=True)
filtered_closures = np.load("gw_filtered_detected_loop_closures.npy", allow_pickle=True)

# Extract indices
orig_i, orig_j = zip(*original_closures)
filt_i, filt_j = zip(*filtered_closures)

# Plot original vs filtered
plt.scatter(orig_i, orig_j, s=1, label="Lpgw Closures (GW)", alpha=0.5, color='red')
plt.scatter(filt_i, filt_j, s=1, label="GW Closures (GW)", alpha=0.5, color='blue')

# Set axis labels and legend
plt.xlabel("Scan Index i")
plt.ylabel("Scan Index j")
plt.legend()

# Set dynamic axis limits
plt.xlim(0, max(max(orig_i), max(filt_i)))
plt.ylim(0, max(max(orig_j), max(filt_j)))

# Set title
plt.title("Comparison of LPGW vs GW Loop Closures (GW)")

# Show the plot
plt.show()
