import torch


def pdlmc_step(x, lam, eta, potential_f, constraint_g):
    x = x.detach().requires_grad_(True)
    lam = lam.detach()

    potential = potential_f(x) + lam * constraint_g(x)

    grad = torch.autograd.grad(potential, x)[0]

    with torch.no_grad():
        noise = torch.randn_like(x)

        x_new = x - eta * grad + (2 * eta) ** 0.5 * noise

        lam_new = torch.relu(lam + eta * constraint_g(x))

    return x_new, lam_new


def pdlmc_chain(x0, lam0, eta, potential_f, constraint_g, n_steps):
    x = x0
    lam = lam0

    x_chain = [x]
    lam_chain = [lam]

    for _ in range(n_steps):
        x, lam = pdlmc_step(x, lam, eta, potential_f, constraint_g)
        x_chain.append(x)
        lam_chain.append(lam)

    return torch.stack(x_chain), torch.stack(lam_chain)
