import numpy as np
import torch
from torch import FloatTensor


def gather_expert_data(env, expert, num_steps):
    exp_rwd_iter = []
    exp_obs = []
    exp_acts = []

    steps = 0
    while steps < num_steps:
        ep_obs = []
        ep_rwds = []

        t = 0
        done = False

        ob = env.reset()
        ob = ob[0]

        while not done and steps < num_steps:
            act = expert.act(ob)

            ep_obs.append(ob)
            exp_obs.append(ob)
            exp_acts.append(act)

            # TODO: handle truncated case?
            ob, rwd, done, trnc, info = env.step(act)

            ep_rwds.append(rwd)

            t += 1
            steps += 1

        if done or steps == num_steps:
            exp_rwd_iter.append(np.sum(ep_rwds))

    return exp_obs, exp_acts, exp_rwd_iter


def process_traj_data(expert_data):
    exp_rwd_iter = expert_data['rews']
    exp_obs = expert_data['obs']
    exp_acts = expert_data['acs']

    temp_obs = []
    temp_acts = []
    for i in range(exp_rwd_iter.shape[0]):
        temp_obs.extend(exp_obs[i])
        temp_acts.extend(exp_acts[i])
        exp_rwd_iter[i] = np.sum(exp_rwd_iter[i])

    exp_obs = np.array(temp_obs)
    exp_acts = np.array(temp_acts)

    return {"rews": exp_rwd_iter, "obs": exp_obs, "acs": exp_acts}

def bootstrap_expert_data(data, num_bags):    
    # Get the lengths of the arrays
    length = len(data["rews"])
    
    num_bags = min(length, num_bags)
    # Generate random indices for shuffling
    indices = np.arange(length)
    np.random.shuffle(indices)
    
    # Split the indices into n_subsets
    subset_indices = np.array_split(indices, num_bags)
    
    # Initialize lists to store subsets
    subsets = []
    
    # Create subsets using the indices
    for subset_idx in subset_indices:
        subset = {}
        for key, value in data.items():
            subset[key] = [value[i] for i in subset_idx]
        subsets.append(subset)
    
    return subsets


def get_flat_grads(f, net):
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads


def get_flat_params(net):
    return torch.cat([param.view(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
    g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10,
    success_ratio=0.1
):
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio \
            and actual_improv > 0 \
                and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params
