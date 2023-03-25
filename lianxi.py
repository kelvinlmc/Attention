import torch
n_train=50
x_train, _ = torch.sort(torch.rand(n_train) * 5)
X_tile = x_train.repeat((n_train, 1))
keys = X_tile.reshape((n_train, -1))
print(x_train,x_train.size())
print(X_tile,X_tile.size())
print(keys,keys.size())


