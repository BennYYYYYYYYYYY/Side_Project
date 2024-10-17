# linear 用法
from torch import nn
import torch
from torch.nn import functional as F
inputs = torch.randn(100, 256) # 生成常態分布, 100個樣本, 每個樣本有256個特徵
weight = torch.randn(20, 256) 
x = F.linear(inputs, weight) # F.linear(): 參數有輸入、權重, 權重圍初始值, 會不斷更新

# 將squential model 改寫 functional API
inputs = torch.randn(100, 256)
x = nn.Linear(256, 20)(inputs) # (inputs): 表示立即把inputs資料丟入模型
x = F.relu(x)
x = nn.Linear(20, 10)(x)
x = F.relu(x)
x = F.softmax(x, dim=1)


# 使用類別定義模型
class Net(nn.Module): # 定義Net, Net繼承自nn.Module
    def __init__(self): # 初始化
        super(Net, self).__init__()
        # super(): 用來調用父類(nn.Module)的初始化
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.flatten(x, 1) # 從二維開始攤平
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        output = F.softmax(x, dim=1)
        return output

from torchinfo import summary    
model =Net()
summary(model, (1, 28, 28))


# 使用類別定義模型, 進行手寫阿拉伯MNIST辨識
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = "" # 預設路徑
BATCH_SIZE = 1024  # 批量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

train_ds = MNIST(PATH_DATASETS, train=True, download=True, # 訓練資料
                 transform=transforms.ToTensor())

test_ds = MNIST(PATH_DATASETS, train=False, download=True,  # 測試資料
                 transform=transforms.ToTensor())

# 建立模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x 
    
model = Net().to(device) # 建立實體

# 訓練模型
epochs = 5
lr = 0.1

train_loader = DataLoader(train_ds, batch_size=600) # 建立train的DataLoader
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 設定優化器
criterion = nn.CrossEntropyLoss() # 損失函數

model.train() # 訓練模式
loss_list=[]
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # 計算損失(loss)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * batch_idx / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')

import matplotlib.pyplot as plt
plt.plot(loss_list, 'r')
plt.show()


test_loader = DataLoader(test_ds, batch_size=600)

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset) # 平均損失
batch = batch_idx * len(data) 
data_count = len(test_loader.dataset)
percentage = 100. * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count}' + 
      f' ({percentage:.0f}%)\n')

predictions = []
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i][0], test_ds[i][1]
        data = data.reshape(1, *data.shape).to(device)
        output = torch.argmax(model(data), axis=-1)
        predictions.append(str(output.item()))

print('actual    :', test_ds.targets[0:20].numpy())
print('prediction: ', ' '.join(predictions[0:20]))
from skimage import io
from skimage.transform import resize
import numpy as np

for i in range(10):
    uploaded_file = f'C:\\Users\\user\\Desktop\\Python\\pytorch data\\myDigits\\{i}.png'
    image1 = io.imread(uploaded_file, as_gray=True)

    image_resized = resize(image1, (28, 28), anti_aliasing=True)    
    X1 = image_resized.reshape(1,28, 28) #/ 255.0

    X1 = torch.FloatTensor(1-X1).to(device)

    predictions = torch.softmax(model(X1), dim=1)
    print(f'actual/prediction: {i} {np.argmax(predictions.detach().cpu().numpy())}')
        

# 使用另一種損失函數(Negative log Loss), 進行手寫阿拉伯數字辨識
# Negative log Loss: 最小化預測概率分布與真實概率分布之間的差異
    
# 建立模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 256) 
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(256, 10) 
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        output = F.log_softmax(x, dim=1) # 這裡改用 F.log_softmax: 避免損失值為負值
        return output   

# 建立模型物件
model = Net().to(device)

epochs = 5
lr=0.1


train_loader = DataLoader(train_ds, batch_size=600)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
loss_list = []    
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # 計算損失(loss)
        loss = F.nll_loss(output, target) # 使用F.null取代CrossEntropyLoss 才可以使用softmax
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * batch_idx / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')

test_loader = DataLoader(test_ds, batch_size=600)

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # 計算損失和  
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.view_as(pred)).sum().item()


test_loss /= len(test_loader.dataset)
batch = batch_idx * len(data)
data_count = len(test_loader.dataset)
percentage = 100. * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count}' + 
      f' ({percentage:.0f}%)\n')

'''
處理多類別分類問題時，常見的損失函數包括CrossEntropyLoss和Negative Log Likelihood Loss, NLL Loss
NLL Loss: 當使用NLL Loss時，模型輸出需要經過log_softmax處理，即對每個類別的softmax概率取對數。NLL Loss計算的是，模型預測的對數概率與實際標籤之間的負值。
PyTorch的CrossEntropyLoss實際上結合了log_softmax和nll_loss的計算。當你選擇使用log_softmax加上nll_loss的組合時，相當於手動分解了CrossEntropyLoss的計算過程。 
'''