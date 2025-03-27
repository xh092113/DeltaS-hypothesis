import torch
import numpy as np
import torch.autograd.functional as F

def generate_data(n, d, k, z_sample_mode):
    # generate w_star as ground truth, first choose nonzero indices,
    # then assign random values in (-1, 1). For nonzero indices, 
    # we ensure the abs value >= 1e-2, else we set it to 1e-2.
    w_star = torch.zeros(d, dtype=torch.float64)
    nonzero_indices = np.random.choice(d, k, replace=False)
    nonzero_values = np.random.uniform(-1, 1, k) 
    while np.any(np.abs(nonzero_values) < 1e-2):
        nonzero_values = np.random.uniform(-1, 1, k) 
    nonzero_values = torch.tensor(nonzero_values, dtype=torch.float64)
    # nonzero_values[torch.abs(nonzero_values) < 1e-2] = 1e-2
    w_star[nonzero_indices] = nonzero_values 

    # generate zs and ys
    # z_sample_mode = 1 or 2, meaning uniform (pm1)^d of Nd(0, 1)
    # here we don't add noise to y, we add during iteration
    if z_sample_mode == 1:
        zs = torch.randint(0, 2, (n, d), dtype=torch.float64) * 2 - 1
    elif z_sample_mode == 2:
        zs = torch.randn(n, d, dtype=torch.float64)
    else:
        raise ValueError("Invalid z_sample_mode")
    ys = zs @ w_star 

    return w_star, zs, ys