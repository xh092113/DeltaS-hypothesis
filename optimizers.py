import torch
import random
import torch.autograd.functional as F

## Loss function. y is generated using <w*, z> + noise, 
## and we use w = u^2 - v^2 to fit w*, where x = [u, v],
## and the loss is (0.5 * (<w, z> - y)^2).mean()

def loss_function(x, zs, ys):
    ## choose random cuda from 0 ~ 3
    # cuda_id = random.randint(0, 3)
    # device = torch.device(f"cuda:{cuda_id}") 
    ## view num of gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    zs = zs.to(device)
    ys = ys.to(device)

    ## cut x into u and v
    d = x.shape[0] // 2
    u, v = x[:d], x[d:]
    ## ys have been added with label noise (coordinate-wise ± delta)
    w = u * u - v * v

    loss = (0.5 * (w @ zs.T - ys) ** 2).mean() 
    loss.to("cpu")

    return loss

## Adam optimizer.
## Calculation of H implemented using F.hessian.
## Note that tr(PHP) = tr((diagH)^1/2) in this setting.
def adam(task_prefix, init_x, zs, ys, delta=1e-2, learning_rate=0.1, \
         beta1=0.95, beta2=0.99, epsilon=1e-6, num_iterations=1000,
         track_H_trace=False,
         track_PHP_trace=False,
         exponent=0.5):
    
    # print("Adam started. init_x:", init_x, "lr:", learning_rate, \
    #        "beta1:", beta1, "beta2:", beta2, \
    #        "epsilon:", epsilon, "num_iterations:", num_iterations,
    #        "track_H_trace:", track_H_trace,
    #        "track_PHP_trace:", track_PHP_trace)
    
    x = init_x.clone().detach().requires_grad_(True).double()
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

    x_trajectory = []
    loss_trajectory = []
    H_trace_trajectory = []
    PHP_trace_trajectory = []
    v_trajectory = []

    for current_iter in range(num_iterations):
        if current_iter % 100000 == 0:
            print(f"adam task {task_prefix} iter {current_iter}")
        y_noise = torch.randint(0, 2, ys.shape, device=ys.device, dtype=ys.dtype) * 2 * delta - delta
        loss = loss_function(x, zs, ys + y_noise)
        loss.backward()
        g = x.grad

        if track_H_trace or track_PHP_trace:
            H = F.hessian(lambda x: loss_function(x, zs, ys), x) ## Clean Hessian
            diag_H = torch.diag(H)
            H_trace = torch.trace(H).item()
            ## negative values may exist, since we are not strictly on the manifold
            diag_H_clamped = torch.clamp(diag_H, min=0) 
            PHP_trace = torch.sum(torch.sqrt(diag_H_clamped)).item()
            if track_H_trace:
                H_trace_trajectory.append(H_trace)
            if track_PHP_trace:
                PHP_trace_trajectory.append(PHP_trace)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        v_trajectory.append(torch.sum(v).item())
        m_hat, v_hat = m, v  # disable bias correction
        u = m_hat / (torch.pow(v_hat, exponent) + epsilon)

        with torch.no_grad():
            x -= learning_rate * u
        x.grad.zero_()
        x_trajectory.append(x.detach().clone())
        loss_trajectory.append(loss.item())

    return x_trajectory, loss_trajectory, H_trace_trajectory, PHP_trace_trajectory, v_trajectory


## manual calculation of clean gradient and clean hessian.
## NOTE: this function is correct iff iteration is near manifold.
def loss_manual(x, zs, ys):
    d = x.shape[0] // 2
    n = zs.shape[0]
    u, v = x[:d], x[d:]
    loss = 0
    gradient = torch.zeros(2 * d, dtype=torch.float64)
    hessian = torch.zeros(2 * d, 2 * d, dtype=torch.float64)
    for i in range(n):
        z, y = zs[i], ys[i]
        l = (z * (u**2-v**2)).sum() - y
        loss += 0.5 * l**2
        zu = (z * u).reshape(-1)
        zv = (z * v).reshape(-1)
        gradient[:d] += 2 * l * zu
        gradient[d:] -= 2 * l * zv
        hessian[:d, :d] += 4 * zu.reshape(-1, 1) @ zu.reshape(1, -1)
        hessian[d:, d:] += 4 * zv.reshape(-1, 1) @ zv.reshape(1, -1)
        hessian[:d, d:] -= 4 * zu.reshape(-1, 1) @ zv.reshape(1, -1)
        hessian[d:, :d] -= 4 * zv.reshape(-1, 1) @ zu.reshape(1, -1)
    loss /= n
    gradient /= n
    hessian /= n
    return loss, gradient, hessian


## Adam optimizer.
## Calculation of H implemented manually.
## Note that tr(PHP) = tr((diagH)^1/2) in this setting.
def adam_manual(init_x, zs, ys, delta=1e-2, learning_rate=0.1, \
         beta1=0.95, beta2=0.99, epsilon=1e-6, num_iterations=1000,
         track_H_trace=False,
         track_PHP_trace=False,
         exponent=0.5):
    
    print("Adam started. init_x:", init_x, "lr:", learning_rate, \
           "beta1:", beta1, "beta2:", beta2, \
           "epsilon:", epsilon, "num_iterations:", num_iterations,
           "track_H_trace:", track_H_trace,
           "track_PHP_trace:", track_PHP_trace)
    
    x = init_x.clone().detach().requires_grad_(True).double()
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

    x_trajectory = []
    loss_trajectory = []
    H_trace_trajectory = []
    PHP_trace_trajectory = []
    v_trajectory = []

    for current_iter in range(num_iterations):
        if current_iter % 100 == 0:
            print("iter:", current_iter)
        y_noise = torch.randint(0, 2, ys.shape, device=ys.device, dtype=ys.dtype) * 2 * delta - delta
        loss = loss_function(x, zs, ys + y_noise)
        loss.backward()
        g = x.grad

        if track_H_trace or track_PHP_trace:
            _loss, _gradient, H = loss_manual(x, zs, ys)
            diag_H = torch.diag(H)
            H_trace = torch.trace(H).item()
            PHP_trace = torch.sum(torch.sqrt(diag_H)).item()
            if current_iter % 100 == 0:
                # print("_gradient:", _gradient)
                # print("diag_H:", diag_H)
                print("H_trace:", H_trace, "PHP_trace:", PHP_trace)
            if track_H_trace:
                H_trace_trajectory.append(H_trace)
            if track_PHP_trace:
                PHP_trace_trajectory.append(PHP_trace)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        # if current_iter % 100 == 0:
            # print("1-norm of v:", torch.sum(v).item())
        v_trajectory.append(torch.sum(v).item())
        m_hat, v_hat = m, v  # disable bias correction
        u = m_hat / (torch.pow(v_hat, exponent) + epsilon)

        with torch.no_grad():
            x -= learning_rate * u
        x.grad.zero_()
        x_trajectory.append(x.detach().clone())
        loss_trajectory.append(loss.item())

    return x_trajectory, loss_trajectory, H_trace_trajectory, PHP_trace_trajectory, v_trajectory


## SGD optimizer.
def sgd(task_prefix, init_x, zs, ys, delta=1e-2, learning_rate=0.1, num_iterations=1000):
    x = init_x.clone().detach().requires_grad_(True).double()
    x_trajectory = []
    loss_trajectory = []
    # hessian_trace_trajectory = []

    for current_iter in range(num_iterations):
        y_noise = torch.randint(0, 2, ys.shape, device=ys.device, dtype=ys.dtype) * 2 * delta - delta
        loss = loss_function(x, zs, ys + y_noise)
        loss.backward()
        # H = F.hessian(lambda x: loss_function(x, zs, ys + y_noise), x)
        # H_trace = torch.trace(H)
        # hessian_trace_trajectory.append(H_trace.item())
        g = x.grad

        ## DELETE THIS
        if current_iter % 100000 == 0:
            print(f"sgd task {task_prefix} iter {current_iter}")
            # print("sgd Hessian Trace:", H_trace)
        # if current_iter == 0:
        #     print("gradient:", g)
        #     print("Hessian:", H)
        #     print("Gradient:", g)

        with torch.no_grad():
            x -= learning_rate * g
        x.grad.zero_()
        x_trajectory.append(x.detach().clone())
        loss_trajectory.append(loss.item())

    return x_trajectory, loss_trajectory