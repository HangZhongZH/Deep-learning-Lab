import torch
from random import randrange

def softmax_regression_loss_grad(theta, x, y):
    SampleNum, _ = x.shape
    FeaNum = theta.shape[1]
    grad = torch.zeros(theta.shape)
    for k in range(FeaNum):

        theta_now = theta[:, k].reshape(-1, 1)
        for m in range(SampleNum):
            sample_now = x[m, :].reshape(1, -1)
            p_num = torch.exp(torch.mm(sample_now, theta_now))
            p_den = 0
            for j in range(FeaNum):
                theta_temp = theta[:, j].reshape(-1, 1)
                p_den += torch.exp(torch.mm(sample_now, theta_temp))
            p = (p_num / p_den)
            p = p.item()
            if y[m] == k:
                y_true = 1
            else:
                y_true = 0
            #print(x[m, :].shape, (y_true - p).shape)
            grad[:, k] += -x[m, :] * (y_true - p)
    return grad

def softmax_regression_loss(theta, x, y):
    SampleNum, _ = x.shape
    FeaNum = theta.shape[1]
    loss = 0
    for i in range(SampleNum):
        sample_now = x[i, :].reshape(1, -1)
        p_den = 0
        y_true = y[i].item()
        for j in range(FeaNum):
            p_den_temp = torch.exp(torch.mm(sample_now, theta[:, j].reshape(-1, 1)))
            p_den += p_den_temp
        for k in range(FeaNum):
            theta_now = theta[:, k].reshape(-1, 1)
            p_num = torch.exp(torch.mm(sample_now, theta_now))
            p = p_num / p_den
            #p = p.item()
            if y_true == k:
                y_true1 = 1
            else:
                y_true1 = 0
            loss_temp = -y_true1 * torch.log(p)
            loss += loss_temp
    return loss

def grad_check(f, x, grad_get, num_checks=10, epsilon=1e-6):
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

NumClasses = 10
FeaDim = 20
NumItems = 100
theta = torch.randn(FeaDim, NumClasses)
x = torch.randn(NumItems, FeaDim)
y = torch.randint(0, NumClasses, (NumItems, 1))

grad = softmax_regression_loss_grad(theta, x, y)
#print(grad)
grad_checked = grad_check(lambda th: softmax_regression_loss(th, x, y), theta, grad)
print(grad_checked)
