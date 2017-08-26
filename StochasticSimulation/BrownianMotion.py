import numpy as np
import matplotlib.pyplot as plt

def simulate_wiener_path(num_paths, path_length, T=1):
    dt = T / path_length
    dW = np.sqrt(dt) * np.random.randn(num_paths, path_length)
    W = np.cumsum(dW, axis=1)
    return W

# Simulate 1000 paths of the wiener process and calculate the mean
sample_paths = simulate_wiener_path(1000, 200)
mean_path = np.mean(sample_paths, axis=0)

# Plot the mean and the first 100 paths
plt.figure(figsize=(12, 7))
_ = plt.plot(sample_paths.T[:, :100])
_ = plt.plot(mean_path, color='black', lw=3)

plt.show()