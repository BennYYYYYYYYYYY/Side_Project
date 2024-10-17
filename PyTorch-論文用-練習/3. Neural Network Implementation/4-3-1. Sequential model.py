# Squential model 兩種寫法

import torch
from torch import nn
from torch.nn import functional as F

model = nn.Sequential(
    nn.Linear(256, 20),
    nn.ReLU(),
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Softmax(dim=1)
)

'''
可以為每一神經層命名, 使用字典(OrderedDict)資料結構, 設定名稱及神經層種類, 逐一配對
'''
from collections import OrderedDict # OrderedDict 會記住鍵值的順序
model = nn.Sequential(OrderedDict([
    ('linear1', nn.Linear(256, 20)),
    ('relu1', nn.ReLU()),
    ('linear2', nn.Linear(20, 64)),
    ('relu2', nn.ReLU()),
    ('softmax', nn.Softmax(dim=1))
]))

# 顯示模型結構
from torchinfo import summary 
summary(model, (1, 256)) # (1, 256): 指定傳入模型的輸入數據形狀, 1: batch size, 256: 特徵數量(輸入大小) 

