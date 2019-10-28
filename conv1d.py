import numpy as np

import data_conv

res = np.zeros((1000, 256))

inp = np.pad(data_conv.input, [(4,), (0,)])

k_size = 9
k_num = 256
f_l = 1000

for k_i in range(k_num):
    for f_i in range(f_l):
        res[f_i][k_i] = np.sum(inp[f_i:f_i + k_size] * data_conv.weights[:, :, k_i]) + data_conv.bias[k_i]
