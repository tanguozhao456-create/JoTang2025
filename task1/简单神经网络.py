import torch
from torch import Tensor, nn
from torch._prims import RETURN_TYPE
num_inputs = 3
num_hiddens1 = 2
num_hiddens2 = 4
num_outputs = 3
num_nets = 3
x = nn.Parameter(torch.tensor([[1, 5, -9],[2, -4, 10],[3, 5, 11]], dtype=torch.float32))
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens1, requires_grad=True) * 0.01)
W2 = nn.Parameter(torch.randn(num_hiddens1, num_hiddens2, requires_grad=True) * 0.01)
W3 = nn.Parameter(torch.randn(num_hiddens2, num_outputs, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens1, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_hiddens2, requires_grad=True))
b3 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

W = [W1, W2, W3]
b = [b1, b2, b3]

def relu(X):
    M = torch.zeros_like(X)
    return torch.max(X, M)

def net(X):
    Z = X
    for i in range(0, num_nets-1):
        Z = Z @ W[i]+b[i]
        Z = relu(Z)
    Y = Z @ W[num_nets-1] + b[num_nets-1]
    return Y

Y = net(x)
print(Y)