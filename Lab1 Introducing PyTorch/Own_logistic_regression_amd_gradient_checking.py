import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from random import randrange

def logistic_regression_loss_grad(theta, x, y):
    grad = -torch.mm(x.t(), y - torch.sigmoid(torch.mm(x, theta)))
    return grad

theta = torch.zeros(1, 1)
x = torch.Tensor([[1]])
y = torch.Tensor([[0]])
grad = logistic_regression_loss_grad(theta, x, y)
print(grad)

x, y = tuple(torch.Tensor(z) for z in load_digits(2, True))
x = torch.cat((x, torch.ones(x.shape[0], 1)), 1)
y = y.reshape(-1, 1)
perm = torch.randperm(y.shape[0])
x_train = x[perm[: 260], :]
y_train = y[perm[: 260], :]
x_test = x[perm[260: ], :]
y_test = y[perm[260: ], :]

rate = .001
theta_gd = torch.rand(x.shape[1], 1)
for i in range(10):
    grad = logistic_regression_loss_grad(theta_gd, x_train, y_train)
    theta_gd = theta_gd - rate * grad
    loss = torch.nn.functional.binary_cross_entropy_with_logits(torch.mm(x_train, theta_gd), y_train)
    print('Train loss is', loss)
print('Best theta is', theta_gd)
test_error = torch.nn.functional.binary_cross_entropy_with_logits(torch.mm(x_test, theta_gd), y_test)
print('Test error is', test_error)


#Gradient checking
def grad_check(f, x, grad_get, num_checks=10, epsilon=1e-4):
    sum_error = 0
    for i in range(num_checks):
        idx = tuple([randrange(m) for m in x.shape])
        oldval = x[idx].item()
        x[idx] = oldval + epsilon
        fxp = f(x)
        x[idx] = oldval - epsilon
        fxm = f(x)
        x[idx] = oldval

        grad_numerical = (fxp - fxm) / (2 * epsilon)
        grad_analy = grad_get[idx]
        rel_error = (grad_numerical - grad_analy)**2 /((grad_analy) + (grad_numerical) + 1e-7)**2
        sum_error += rel_error
    return sum_error/num_checks

theta = torch.rand_like(theta_gd) * .001
grad = logistic_regression_loss_grad(theta, x_test, y_test)

def func(th):
    sigm = torch.sigmoid(torch.mm(x_test, th))
    f = -torch.mm(y_test.t(), torch.log(sigm)) - torch.mm((1 - y_test.t()), torch.log(1 - sigm))
    return f

relerr = grad_check(func, theta, grad)
print('Average error', relerr)