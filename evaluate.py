import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from optimizers import loss_function, adam, sgd
from datasets import generate_data


def suboptimality(w_star, x):
    d = w_star.shape[0]
    u, v = x[:d], x[d:]
    w = u * u - v * v
    return torch.norm(w - w_star) ** 2


def evaluate_adam(task_prefix, n, init_x, w_star, zs, ys, train_N, delta, lr, beta_1, beta_2, epsilon, exponent):
    num_iterations = 10000000 ## originally 10^7
    
    zs_test = zs[train_N:]
    ys_test = ys[train_N:]
    zs = zs[:n]
    ys = ys[:n]
    xs, train_losses_noised, _1, _2, _3 = adam(init_x, zs, ys, delta=delta, \
                        learning_rate=lr, beta1=beta_1, beta2=beta_2, \
                        epsilon=epsilon, num_iterations=num_iterations, \
                            exponent=exponent)
    train_losses_clean = []
    test_losses_clean = []
    # suboptimalities = []
    print_indexes = range(0, num_iterations, 1000)
    for i in print_indexes:
        train_losses_clean.append(loss_function(xs[i], zs, ys).item())
        test_losses_clean.append(loss_function(xs[i], zs_test, ys_test).item())
        # suboptimalities.append(suboptimality(w_star, xs[i]).item())

    # plot the results, with x-axis as n, y-axis as loss
    # we use orange for testloss, and purple for trainloss
    plt.figure(figsize=(10, 10))
    plt.plot(print_indexes, test_losses_clean, label='Test Loss', color='orange')
    plt.plot(print_indexes, train_losses_clean, label='Train Loss', color='purple')
    # plt.plot(print_indexes, suboptimalities, label='Suboptimality', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss Trajectory of Adam with exponent={}'.format(exponent))
    plt.legend()
    plt.savefig(f'results/{task_prefix}/loss.png')
    plt.close()
    
    # save test and train losses to a file using numpy.save
    np.save(f'results/{task_prefix}/test.npy', test_losses_clean)
    np.save(f'results/{task_prefix}/train.npy', train_losses_clean)

    return np.mean(test_losses_clean[-1000:])


def evaluate_sgd(task_prefix, n, init_x, w_star, zs, ys, train_N, delta, lr):
    num_iterations = 10000000 ## originally 10^7

    zs_test = zs[train_N:]
    ys_test = ys[train_N:]
    zs = zs[:n]
    ys = ys[:n]
    xs, train_losses_noised = sgd(task_prefix, init_x, zs, ys, delta=delta, \
                        learning_rate=lr, num_iterations=num_iterations)
    
    train_losses_clean = []
    test_losses_clean = []
    # suboptimalities = []
    print_indexes = range(0, num_iterations, 1000)
    for i in print_indexes:
        train_losses_clean.append(loss_function(xs[i], zs, ys).item())
        test_losses_clean.append(loss_function(xs[i], zs_test, ys_test).item())
        # suboptimalities.append(suboptimality(w_star, xs[i]).item())

    ## plot the results, with x-axis as n, y-axis as loss
    ## we use orange for testloss, and purple for trainloss
    plt.figure(figsize=(10, 10))
    plt.plot(print_indexes, test_losses_clean, label='Test Loss', color='orange')
    plt.plot(print_indexes, train_losses_clean, label='Train Loss', color='purple')
    # plt.plot(print_indexes, suboptimalities, label='Suboptimality', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss Trajectory of SGD')
    plt.legend()
    plt.savefig(f"results/{task_prefix}/loss.png")    
    plt.close()

    # save test and train losses to a file using numpy.save
    np.save(f"results/{task_prefix}/test.npy", test_losses_clean)
    np.save(f"results/{task_prefix}/train.npy", train_losses_clean)

    return np.mean(test_losses_clean[-1000:])
