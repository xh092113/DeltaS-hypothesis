import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datasets import generate_data
from evaluate import evaluate_sgd


def main(args):
    torch.manual_seed(66)
    np.random.seed(66)

    ## parameters setup
    train_N = args.n
    d, k, delta = args.d, args.k, args.delta
    z_sample_mode = args.z_sample_mode
    lrs = args.lrs
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

    ## result directory: results/sgd/{d,k,delta,lr}/{n}
    task_prefix = "sgd/" + f"d{d}-k{k}-delta{delta}-lrs{lrs}/" + f"n{train_N}"
    os.makedirs("results/" + task_prefix, exist_ok=True)

    evaluate_sgd(task_prefix, train_N, init_x, w_star, zs, ys, train_N, delta, lrs)
    return

    # plot the results, with x-axis as n, y-axis as loss
    # we use orange for sgd, and different shades of 
    # purple for adam with different exponents

    ## save the results to a file using numpy.save
    np.save(f'results2/losses/losssgd.npy', losssgd)
    np.save(f'results2/losses/lossadam.npy', lossadam)

    plt.figure(figsize=(10, 10))
    # plt.xscale('log')
    plt.plot(test_ns, losssgd, label='SGD', color='orange')
    plt.plot(test_ns, lossadam, label='Adam', color='purple')
    # for exp in adam_test_exponents:
        # plt.plot(test_ns, lossadam[exp], label='Adam with exp={}'.format(exp), \
                #  color='purple', alpha=0.5+0.5*exp)
    plt.xlabel('n')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs. n Trajectories of SGD and Adam')
    plt.legend()
    plt.savefig(f'results2/results2-d{d}-k{k}-e{e}.png')
    plt.close()

    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SGD experiment with specified parameters.")
    parser.add_argument('--d', type=int, help='Dimension of z', default=int(1e3)) # original 1e4
    parser.add_argument('--z_sample_mode', type=int, choices=[1, 2], 
                        help='Sampling mode for z: 1 for uniform, 2 for normal', default=1)
    parser.add_argument('--k', type=int, help='Sparsity of w* (must be <= d)', default=3)
    parser.add_argument('--delta', type=float, help='Scale of noise added to y', default=0.5)
    parser.add_argument('--lrs', type=float, help='Learning rate for Sgd', default=0.005)
    parser.add_argument('--n', type=int, help='Number of training samples', default=50) 
    # parser.add_argument('--lra', type=float, help='Learning rate for Adam', default=0.1)
    # parser.add_argument('--beta1', type=float, help='Beta1 for Adam', default=0.9)
    # parser.add_argument('--beta2', type=float, help='Beta2 for Adam', default=0.95) # 0.999 or 0.95, usually used
    # parser.add_argument('--epsilon', type=float, help='Epsilon for Adam', default=1e-6)
    args = parser.parse_args()

    if args.k > args.d:
        raise ValueError("k must be less than or equal to d.")
    print(args, flush=True)
    main(args)
