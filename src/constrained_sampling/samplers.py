import torch


def pdlmc_step(x, lam, eta, potential_f, constraint_g):
    x = x.detach().requires_grad_(True)
    lam = lam.detach()

    potential = potential_f(x) + lam * constraint_g(x)

    grad = torch.autograd.grad(potential, x)[0]

    with torch.no_grad():
        noise = torch.randn_like(x)

        x_new = x - eta * grad + torch.sqrt(2 * eta) * noise

        lam_new = torch.relu(lam + eta * constraint_g(x))

    return x_new, lam_new
