import torch


def pdlmc_step(x, lam, eta, potential_f, constraint_g):
    x = x.detach().requires_grad_(True)
    lam = lam.detach()

    g = constraint_g(x)
    potential = potential_f(x) + lam * g

    grad = torch.autograd.grad(potential, x)[0]

    with torch.no_grad():
        noise = torch.randn_like(x)

        x_new = x - eta * grad + (2 * eta) ** 0.5 * noise

        lam_new = torch.relu(lam + eta * g)

    return x_new, lam_new


def pdlmc_chain(x0, lam0, eta, potential_f, constraint_g, n_steps):
    x = x0.detach().clone()
    lam = lam0.detach().clone()

    x_chain = [x]
    lam_chain = [lam]

    for _ in range(n_steps):
        x, lam = pdlmc_step(x, lam, eta, potential_f, constraint_g)
        x_chain.append(x)
        lam_chain.append(lam)

    return torch.stack(x_chain), torch.stack(lam_chain)


# Here the rejection sampling was was implemented just for the normalized gaussian, for the sake of simplicity
def rejection_sampling(num_samples, center, constraint_r, batch_size=100000):
    accepted = []
    n_accepted = 0
    n_proposed = 0

    with torch.no_grad():
        while n_accepted < num_samples:
            x_proposed = torch.randn(batch_size, 2) + center
            r_values = constraint_r(x_proposed)
            mask = r_values <= 1.0

            accepted_samples = x_proposed[mask]
            accepted.append(accepted_samples)

            n_accepted += accepted_samples.shape[0]
            n_proposed += batch_size

    return torch.cat(accepted, dim=0)[:num_samples], n_accepted / n_proposed
