import torch
import matplotlib.pyplot as plt

N = 1000
theta_true = torch.Tensor([[1.5], [2.]])

x = torch.rand(N, 2)
x[:, 1] = 1.
y_true = torch.mm(x, theta_true) + 0.1 * torch.randn(N, 1)

plt.scatter(x[:, 0].numpy(), y_true.numpy())
#plt.show()

x_inv = torch.pinverse(x)
theta_pinv = torch.mm(x_inv, y_true)
print(theta_pinv)

u, s, v = torch.svd(x)
ut = u.t()
st = torch.diag(s**(-1))
x_pinv_svd = torch.mm(torch.mm(v, st), ut)
theta_pinv_svd = torch.mm(x_pinv_svd, y_true)
print(theta_pinv_svd)

def linear_regression_loss_grad(theta, x, y):
    grad = -(1/x.shape[0]) * torch.mm(x.t(), (y - torch.mm(x, theta)))
    return grad

rate = .1
theta = torch.Tensor([[0], [1]])
for e in range(2000):
    grad = linear_regression_loss_grad(theta, x, y_true)
    theta -= rate * grad
print(theta)




from sklearn.datasets import load_boston
x, y = tuple(torch.Tensor(z) for z in load_boston(True))
x = x[:, [2, 5]]
x = torch.cat((x, torch.ones((x.shape[0], 1))), 1)
y = y.reshape(-1, 1)

perm = torch.randperm(y.shape[0])
x_train = x[perm[0: 253], :]
y_train = y[perm[0: 253], :]
x_test = x[perm[253:], :]
y_test = y[perm[253:], :]

#Pseudo inverse
x_train_inv = torch.pinverse(x_train)
theta = torch.mm(x_train_inv, y_train)
print(theta)
print('MSE', torch.nn.functional.mse_loss(torch.mm(x_test, theta), y_test))

#Gradient descent
rate = .001
theta_gd = torch.rand((x_train.shape[1], 1))
for i in range(10000):
    grad = linear_regression_loss_grad(theta_gd, x_train, y_train)
    theta_gd = theta_gd - rate * grad

print('Gradient descent', theta_gd)
print('MSE of test data', torch.nn.functional.mse_loss(torch.mm(x_test, theta_gd), y_test))
