import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt



def main(args):
    torch.manual_seed(66)
    np.random.seed(66)

    ## parameters setup
    train_N = args.n
    d, k, delta = args.d, args.k, args.delta
    z_sample_mode = args.z_sample_mode
    beta1, epsilon, e = args.beta1, args.epsilon, args.e
    D = args.D
    test_N = int(1e4)
    proj_matrix = np.random.randn(2 * d, 2)

    for scheme in [1.25, 1.5, 1.75]:
        C = 0.1 ** (1 - scheme) ## let beta2 = 0.9 when lr = 0.1
        for lra in np.logspace(-3, -1, num=10):
            beta2 = 1 - C * lra ** scheme
            ## result directory: results/adam/scheme{scheme}/beta2{beta2}-lra{lr}/n{n}
            task_prefix = "adam/" + f"scheme{scheme}/" + f"beta2{beta2}-lra{lra}/" + f"n{train_N}"
            task_dir = f'results/{task_prefix}'
            os.makedirs(f"results/plots/{task_prefix}", exist_ok=True)

            deltas = np.load(f'{task_dir}/deltas.npy')
            s_array = np.load(f'{task_dir}/s_array.npy')
            gdeltas = np.load(f'{task_dir}/gdeltas.npy')
            convergence_point = 1000

            ## plot the first 200 steps of gdeltas
            plt.figure(figsize=(12, 8))
            # plt.plot(range(convergence_point + 1, len(s_array)), gdeltas, label='gs', color='green')
            plt.plot(range(convergence_point + 1, convergence_point + 1001), gdeltas[:1000], label='gs', color='green')
            plt.xlabel('Iteration')
            plt.ylabel('gs')
            plt.title('gs Trajectory of Adam with exponent={}'.format(e))
            plt.legend()
            plt.savefig(f'results/plots/{task_prefix}/gdeltas.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adam experiment with specified parameters.")
    parser.add_argument('--d', type=int, help='Dimension of z', default=int(100)) # original 1e4
    parser.add_argument('--z_sample_mode', type=int, choices=[1, 2], 
                        help='Sampling mode for z: 1 for uniform, 2 for normal', default=1)
    parser.add_argument('--k', type=int, help='Sparsity of w* (must be <= d)', default=5)
    parser.add_argument('--delta', type=float, help='Scale of noise added to y', default=0.5)
    # parser.add_argument('--lra', type=float, help='Learning rate for Adam', default=0.1)
    parser.add_argument('--beta1', type=float, help='Beta1 for Adam', default=0.9)
    # parser.add_argument('--beta2', type=float, help='Beta2 for Adam', default=0.95) # 0.999 or 0.95, usually used
    parser.add_argument('--epsilon', type=float, help='Epsilon for Adam', default=1e-6)
    parser.add_argument('--n', type=int, help='Number of training samples', default=50) 
    parser.add_argument('--e', type=int, help='exponent of Adam', default=0.5) 
    parser.add_argument('--D', type=int, help='Frequency of recording', default=100)
    args = parser.parse_args()

    if args.k > args.d:
        raise ValueError("k must be less than or equal to d.")
    print(args)
    main(args)