import torch

N = 1000
theta_True = torch.Tensor([[1.5], [2.]])
x = torch.rand(N, 2)
x[:, 1] = 1.
y_True = torch.mm(x, theta_True) + 0.1 * torch.randn(N, 1)
