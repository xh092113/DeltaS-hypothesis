import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datasets import generate_data
from evaluate import evaluate_adam


def main(args):
    torch.manual_seed(66)
    np.random.seed(66)

    ## parameters setup
    train_N = args.n
    d, k, delta = args.d, args.k, args.delta
    z_sample_mode = args.z_sample_mode
    lra, beta1, beta2, epsilon, e = args.lra, args.beta1, args.beta2, args.epsilon, args.e
    test_N = int(1e4)

    ## generate data and train; the former train_N samples are for training,
    ## and the latter test_N samples are for testing
    w_star, zs, ys = generate_data(train_N + test_N, d, k, z_sample_mode)

    # print("w_star:", w_star)
    # print("example of zs:", zs[:5])
    # print("example of ys:", ys[:5])

    ## A trick borrowed from paper 'Kernel and Rich': let u = v in init_x,
    ## to control initial loss to be E[y^2], which is of constant order,
    ## no matter what the scale of d is. Also, we scale the initial x by 0.1
    ## to make initalization smaller, leading to faster generalization.
    init_x = torch.randn(d, dtype=torch.float64)
    init_x = torch.cat((init_x, init_x), 0)
    init_x = init_x * 0.1 

    ## result directory: results/adam/{beta2,lr}/{n}
    task_prefix = "adam/" + f"beta2{beta2}-lra{lra}/" + f"n{train_N}"
    os.makedirs("results/" + task_prefix, exist_ok=True)

    evaluate_adam(task_prefix, train_N, init_x, w_star, zs, ys, train_N, delta, lra, beta1, beta2, epsilon, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adam experiment with specified parameters.")
    parser.add_argument('--d', type=int, help='Dimension of z', default=int(1e3)) # original 1e4
    parser.add_argument('--z_sample_mode', type=int, choices=[1, 2], 
                        help='Sampling mode for z: 1 for uniform, 2 for normal', default=1)
    parser.add_argument('--k', type=int, help='Sparsity of w* (must be <= d)', default=3)
    parser.add_argument('--delta', type=float, help='Scale of noise added to y', default=0.5)
    parser.add_argument('--lra', type=float, help='Learning rate for Adam', default=0.1)
    parser.add_argument('--beta1', type=float, help='Beta1 for Adam', default=0.9)
    parser.add_argument('--beta2', type=float, help='Beta2 for Adam', default=0.95) # 0.999 or 0.95, usually used
    parser.add_argument('--epsilon', type=float, help='Epsilon for Adam', default=1e-6)
    args = parser.parse_args()

    if args.k > args.d:
        raise ValueError("k must be less than or equal to d.")
    print(args)
    main(args)
