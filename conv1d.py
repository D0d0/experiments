import numpy as np

conv1d_kernel = np.array([1, 2])
data = np.array([0, 3, 4, 5])
result = []
for i in range(3):
    print(data[i:i + 2], "*", conv1d_kernel, "=", data[i:i + 2] * conv1d_kernel)
    result.append(np.sum(data[i:i + 2] * conv1d_kernel))
print("Conv1d output", result)
