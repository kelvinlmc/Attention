'''
#此部分为利用注意力机制拟合训练的数据，权值是根据x值来确定的，测试数据和训练数据之间的相似度来确定权值，最后用该权值得到拟合的y值
import torch
from torch import nn
import matplotlib.pyplot as plt

n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本,返回顺序后的tensor以及对应的索引切片
#print(x_train,_)
#print(torch.normal(0.0, 0.5,(2,2)))
def f(x):
    return 2 * torch.sin(x) + x**0.8
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
print(y_train.size())
x_test = torch.arange(0, 5, 0.1)  # 测试样本  生成数字0开始5结束步长0.1
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
plt.plot(x_test,y_truth,'g')
plt.plot(x_train,y_train,'o')
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
#print(X_repeat.size())
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, y_train)
plt.plot(x_test,y_hat,'y')
plt.show()
'''
import matplotlib.pyplot as plt

'''
#随机噪声拟合曲线（效果非常差）
import torch
from torch import nn
import matplotlib.pyplot as plt

x_test = torch.arange(0, 5, 0.1)
n_train = 50  # 训练样本数
def f(x):
    return 2 * torch.sin(x) + x**0.8
y_test = f(x_test)  # 训练样本的输出
n_test = len(x_test)  # 测试样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)
y_train=torch.normal(0.0, 1, (n_train,))
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
#print(X_repeat.size())
attention_weights = nn.functional.softmax(-(X_repeat - x_test)**2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, y_train)
plt.plot(x_train,y_hat,'y')
plt.plot(x_train,y_train,'o')
plt.plot(x_test,y_test,'r')
#plt.show()
'''
'''
#矩阵相乘
import torch
weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((10,2))
q=torch.bmm(weights.unsqueeze(0), values.unsqueeze(0))
print(q.size())'''

import matplotlib.pyplot as plt
import torch
from torch import nn
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True)) #除了卷积全连接什么的有训练参数，要加入额外的参数
        #就用这个函数，同模型一起训练跟新

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))  #确保查询个数等于键值对数，列式个数和键值对
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

x_test = torch.arange(0, 5, 0.1)
n_test = len(x_test)
n_train=50
x_train, _ = torch.sort(torch.rand(n_train) * 5)
def f(x):
    return 2 * torch.sin(x) + x**0.8
y_test = f(x_test)
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))

#train
#这里是设置键值对,是用来训练网络的训练参数
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1)) #维度不同第0维添加1维，将行复制n_train次列复制一次就是不变
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
#keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
keys = X_tile.reshape((n_train, -1))
print(x_train)
print(X_tile)
print(keys)
# values的形状:('n_train'，'n_train'-1)
#alues = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile.reshape((n_train, -1))
#定义网络优化器损失函数
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
#animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
#开始训练
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    #animator.add(epoch + 1, float(l.sum()))


#test
#这里是测试参数并且查看测试结果
# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plt.plot(x_test,y_test,'r')
plt.plot(x_train,y_train,'o')
plt.plot(x_train,y_hat,'g')
plt.show()

#print(keys)
#print(keys.shape[1].reshape((-1, keys.shape[1])))
'''keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)]
print(X_tile)
print(keys)
a=torch.tensor([1,2,3])

c=a[True,True,True]
print("C",c)'''


#print((1 - torch.eye(n_train)).type(torch.bool).reshape((n_train, -1).shape()))
'''print(torch.rand((1,)))
queries=torch.randn(5)
keys=torch.tensor([[1,2]
                   [2,3]])
queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
print(queries)'''
#看net就知道这里是要将输入的维度转变成和键值对一样的维数也就是说先必须和键一样，因为要拿查询q和所有的键来做相似度计算
#前面得到了每一个的权值然后要和v相乘计算所以维度要和v的维度能做矩阵相乘