import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  #全局池化后变成b*c*1*1，后面全连接所以要展平view(b,c)就是reshape成[b,c]
        y = self.fc(y).view(b, c, 1, 1)
        print(y.size())
        return x * y

model=se_block(512)
input=torch.ones(1,512,26,26)
output=model(input)
#print(output.size())
a=torch.rand(1,512,1,1)
b=torch.rand(1,512,26,26)
print(a*b)
'''
x=torch.rand(1,23,28,28)
b, c, _, _ = x.size()
print(b,c,_,_)
a=torch.rand(1,1,4,4)
print(a)
b=torch.rand(1,1,4,4)
print(b)
print(a*b)
'''
