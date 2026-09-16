import torch


def potential_f(x, center):
    return 0.5 * torch.sum((x - center) ** 2, dim=-1)


def constraint_r(x, a, b):
    return (x[..., 0] / a) ** 2 + (x[..., 1] / b) ** 2


def constraint_g(x, epsilon=1e-3):
    return torch.relu(constraint_r(x, a=0.5, b=2) - 1.0) - epsilon
