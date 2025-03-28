import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datasets import generate_data


def loss_function(x, zs, ys):
    ## cut x into u and v
    d = x.shape[0] // 2
    u, v = x[:d], x[d:]
    ## ys have been added with label noise (coordinate-wise ± delta)
    w = u * u - v * v
    loss = (0.5 * (w @ zs.T - ys) ** 2).mean() 
    return loss

def adam(task_prefix, init_x, zs, ys, delta=1e-2, learning_rate=0.1, \
         beta1=0.95, beta2=0.99, epsilon=1e-6, num_iterations=1000,
         exponent=0.5, D=100):
    
    x = init_x.clone().detach().requires_grad_(True).double()
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

    x_trajectory = []
    l_trajectory = []
    l_clean = []
    s_trajectory = []
    g_trajectory = []

    for current_iter in range(num_iterations):
        if current_iter % 100000 == 0:
            print(f"adam task {task_prefix} iter {current_iter}")
        y_noise = torch.randint(0, 2, ys.shape, device=ys.device, dtype=ys.dtype) * 2 * delta - delta
        loss = loss_function(x, zs, ys + y_noise)
        loss.backward()
        g = x.grad

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # if current_iter <= 100:
        #     print("current_iter", current_iter)
        #     print("g:", g)
        #     print("v:", v)

        m_hat, v_hat = m, v  # disable bias correction
        u = m_hat / (torch.pow(v_hat, exponent) + epsilon)
        s = torch.pow(v_hat, -exponent) ## preconditioner

        ## in the first iterates, s will change dramatically, since v moves from 
        ## 0 to a non-zero value, which will cause s to be very large
        ## So we record s from iteration D-1 instead of 0
        if current_iter % D == D - 1:
            x_trajectory.append(x.detach().clone())
            l_trajectory.append(loss.item())
            l_clean.append(loss_function(x, zs, ys).item())
            s_trajectory.append(s.detach().clone())
            g_trajectory.append(g.detach().clone())

        with torch.no_grad():
            x -= learning_rate * u
        x.grad.zero_()

    return x_trajectory, l_trajectory, l_clean, s_trajectory, g_trajectory

def find_convergence_point(losses):
    length = len(losses)
    plateau_length = 0
    eps = 3e-4
    for i in range(1, length):
        if np.abs(losses[i] - losses[i - 1]) < eps:
            plateau_length += 1
        else:
            plateau_length = 0
        if plateau_length >= 50:
            return i
    return length

def estimate_deltas(task_prefix, n, init_x, w_star, zs, ys, train_N, delta, lra, beta1, beta2, epsilon, e, D, proj_matrix):
    num_iterations = 1000000 ## originally 10^7
    zs_test = zs[train_N:]
    ys_test = ys[train_N:]
    zs = zs[:n]
    ys = ys[:n]
    xs, train_losses_noised, train_losses_clean, s_trajectory, g_trajectory = adam(task_prefix, init_x, zs, ys, delta=delta, \
                        learning_rate=lra, beta1=beta1, beta2=beta2, epsilon=epsilon, \
                        num_iterations=num_iterations, exponent=e, D=D)
    print_indexes = range(0, num_iterations, D)
    convergence_point = 1000
    # convergence_point = find_convergence_point(train_losses_clean)
    deltas = []
    s_array = np.array(s_trajectory)
    g_array = np.array(g_trajectory)
    ref_vector = s_array[convergence_point]
    s_clipped = s_array[convergence_point+1:]
    g_clipped = g_array[convergence_point+1:]

    ## delete this
    # print("first items of gs trajectory:")
    # for i in range(100):
    #     print(f"gs_trajectory[{i}]", _gs_trajectory[i])

    # print("first items of clipped s trajectory:")
    # for i in range(convergence_point, convergence_point + 100):
    #     print(f"gs clipped[{i}]", _gs_trajectory[i])

    # raise ValueError("stop here")
    # return
    
    deltas = np.linalg.norm(s_clipped - ref_vector, axis=1)
    # print((s_clipped - ref_vector).shape)
    # print(g_clipped.shape)
    gdeltas = ((s_clipped - ref_vector) * g_clipped).sum(axis=1).reshape(-1)

    np.save(f'results/{task_prefix}/deltas.npy', deltas)
    np.save(f'results/{task_prefix}/s_array.npy', s_array)
    np.save(f'results/{task_prefix}/gdeltas.npy', gdeltas)
    
    ## two subplots: one for loss, one for delta
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.plot(print_indexes, train_losses_clean, label='Train Loss', color='purple')
    plt.axvline(x=convergence_point * D + D - 1, color='red', linestyle='--', label='Convergence Point')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss Trajectory of Adam with exponent={}'.format(e))
    plt.legend()

    ## second plot: deltas
    plt.subplot(2, 2, 2)
    plt.plot(range(convergence_point + 1, len(s_trajectory)), deltas, label='Delta', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('DeltaS')
    plt.title('DeltaS Trajectory of Adam with exponent={}'.format(e))
    plt.legend()

    ## third plot: s_trajectory
    ## we plot the trajectory before converging as gray, and after converging as red
    plt.subplot(2, 2, 3)
    s_projected = s_array @ proj_matrix
    plt.plot(s_projected[:convergence_point, 0], s_projected[:convergence_point, 1], marker='o', markersize=1, linewidth=1, color='gray')
    plt.plot(s_projected[convergence_point:, 0], s_projected[convergence_point:, 1], marker='o', markersize=1, linewidth=1, color='red')
    plt.scatter(s_projected[convergence_point, 0], s_projected[convergence_point, 1], color='orange', label='Convergence Point')
    plt.xlabel('Proj Dim 1')
    plt.ylabel('Proj Dim 2')
    plt.title('2D Random Projection of Trajectory')
    plt.legend()

    ## fourth plot: gdeltas
    plt.subplot(2, 2, 4)
    plt.plot(range(convergence_point + 1, len(s_trajectory)), gdeltas, label='gs', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('gs')
    plt.title('gs Trajectory of Adam with exponent={}'.format(e))
    plt.legend()

    plt.savefig(f'results/{task_prefix}/delta.png')
    plt.close()

    # save test and train losses to a file using numpy.save
    # np.save(f'results/{task_prefix}/test.npy', test_losses_clean)
    np.save(f'results/{task_prefix}/train.npy', train_losses_clean)

    ## process all s_trajectory
    ## first find the point of convergence



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

    ## generate data and train; the former train_N samples are for training,
    ## and the latter test_N samples are for testing
    w_star, zs, ys = generate_data(train_N + test_N, d, k, z_sample_mode)
    init_x = torch.randn(d, dtype=torch.float64)
    init_x = torch.cat((init_x, init_x), 0)
    init_x = init_x * 0.1 

    for scheme in [2.0, 1.0, 1.5, 3.0, 0.5, 0.75, 1.25, 1.75, 2.5, 3.5]:
        C = 0.1 ** (1 - scheme) ## let beta2 = 0.9 when lr = 0.1
        for lra in np.logspace(-3, -1, num=10):
            beta2 = 1 - C * lra ** scheme
            ## result directory: results/adam/scheme{scheme}/beta2{beta2}-lra{lr}/n{n}
            task_prefix = "adam/" + f"scheme{scheme}/" + f"beta2{beta2}-lra{lra}/" + f"n{train_N}"
            os.makedirs("results/" + task_prefix, exist_ok=True)
            estimate_deltas(task_prefix, train_N, init_x, w_star, zs, ys, train_N, delta, lra, beta1, beta2, epsilon, e, D, proj_matrix)


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

    