import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = ''
BATCH_SIZE = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_ds = MNIST(PATH_DATASETS, train=True, download=True, 
                 transform=transforms.ToTensor()) # 載入MNIST手寫阿拉伯資料

test_ds = MNIST(PATH_DATASETS, train=False, download=True, 
                 transform=transforms.ToTensor()) # 載入測試資料

print(train_ds.data.shape, test_ds.data.shape) # 訓練/測試資料的維度

import math
 
def Conv_Width(W, F, P, S):
    return math.floor(((W - F + 2 * P) / S) + 1) # 給一個圖像時，透過函數確定卷積後的輸出有多寬
# W: 輸入圖像的寬度
# F: Kernel寬度
# P: 填充的數量
# S: 步長

def Conv_Output_Volume(W, F, P, S, out):
    return Conv_Width(W, F, P, S) ** 2 * out
# out: 輸出通道數，即卷積層的Kernel數量
# 寬*高(假設相等)
# 寬*高*卷積核的數量 = 總輸出體積
# 計算卷積層的總輸出體積，即卷積操作後得到的特徵圖的數量和大小

def Conv_Parameter_Count(F, C, out):
    return F ** 2 * C * out # 計算卷積層所需的參數數量

def Pool_Width(W, F, P, S): # 計算池化層操作後的輸出寬度
    return Conv_Width(W, F, P, S)

def Pool_Output_Volume(W, F, P, S, filter_count): # 計算池化操作後的總輸出體積
    return Conv_Output_Volume(W, F, P, S, filter_count)

def Pool_Parameter_Count(W, F, S): # 返回池化層的參數數量
    return 0 # 由於pooling 如最大池化和平均池化不涉及學習的權重，所以返回0

# 測試
print(Pool_Width(Conv_Width(32, 3, 1, 1), 2, 0, 2))
# 輸入圖片32x32, 卷積核3x3, 填充1, 步長:1 
# 池化核2x2, 填充0, 步長2
print(Pool_Width(Conv_Width(16, 3, 1, 1), 2, 0, 2))
print(Pool_Width(Conv_Width(8, 3, 1, 1), 2, 0, 2))

def Conv_Pool_Width(W, F, P, S, F2, P2, S2, n):
    for i in range(n):
        W = Pool_Width(Conv_Width(W, F, P, S), F2, P2, S2)
    return W

Conv_Pool_Width(32, 3, 1, 1, 2, 0, 2, 3) # 疊代3次 (n)



# 建立模型
class ConvNet(nn.Module): # 繼承nn.Module
    def __init__(self, num_classes=10): # num_classes=10: 表示輸出的類別數量
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            # 1: 輸入通道數 (灰度圖像為1)
            # 16: 輸出通道數。
            # kernel_size=5: 卷積核的大小為5x5。
            # stride=1: 卷積操作的步長為1。
            # padding=2: 在輸入周圍添加2像素的零填充
            nn.BatchNorm2d(16), # 對16個輸出通道進行批量標準化
            nn.ReLU(), # 激活函數
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), # 輸出通道改為32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes) # 完全連接layer, 將卷積輸出結果展平
        
    def forward(self, x): # data跑model
        out = self.layer1(x) # 經layer 1
        out = self.layer2(out) # layer 2
        out = out.reshape(out.size(0), -1) # 將卷積層的輸出展平，以適合全連接層的輸入
        out = self.fc(out) # 經全連接層
        out = F.log_softmax(out, dim=1) # 進行多類別分類的最後輸出
        return out 

model = ConvNet().to(device)

epochs = 10
lr=0.1

# 建立 DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# 設定優化器(optimizer)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train() # 訓練模式
loss_list = []    
for epoch in range(1, epochs + 1): # 訓練次數
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad() # reset
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward() 
        optimizer.step() # 更新權重

        if (batch_idx+1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx+1) * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * (batch_idx+1) / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')
        # 每10個batch，將當前損加到損失list並打印出當前epoch的進度和損失

import matplotlib.pyplot as plt

plt.plot(loss_list, 'r') # 訓練過程的損失繪圖
plt.show()

test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE)

model.eval() # 評估模式
test_loss = 0
correct = 0
with torch.no_grad(): # 不須計算梯度
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        test_loss += F.nll_loss(output, target).item() # 計算當前批次的 Negative Log Likelihood Loss
        
        output = model(data) # 預測結果
        
        _, predicted = torch.max(output.data, 1) # torch.max 返回的是每行的最大值及其索引，找出每個樣本最可能的類別
        correct += (predicted == target).sum().item() # 計算正確數量

test_loss /= len(test_loader.dataset) # 累積的損失除以批次的數量，得到平均損失

batch = batch_idx * len(data) # 獲取測試集的總樣本數
data_count = len(test_loader.dataset) # 計算正確預測的百分比
percentage = 100. * correct / data_count 
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count}' + 
      f' ({percentage:.2f}%)\n')

# 實際預測 20 筆資料
predictions = []
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i][0], test_ds[i][1]
        data = data.reshape(1, *data.shape).to(device)
        output = torch.argmax(model(data), axis=-1)
        predictions.append(str(output.item()))

# 比對
print('actual    :', test_ds.targets[0:20].numpy())
print('prediction: ', ' '.join(predictions[0:20]))

import numpy as np

i=18 # 選取第19筆樣本
data = test_ds[i][0] # 選取數據集第19個樣本的圖像資料
data = data.reshape(1, *data.shape).to(device) # 增加一維為1
print(data.shape)
predictions = torch.softmax(model(data), dim=1) # 沿著二維(類別), 進行機率分配
print(f'0~9預測機率: {np.around(predictions.cpu().detach().numpy(), 2)}') 
print(f'0~9預測機率: {np.argmax(predictions.cpu().detach().numpy(), axis=-1)}') # axis = -1 為對最後一個維度進行操作


X2 = test_ds[i][0] 
plt.imshow(X2.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show()  

test_ds[i][0]

# 模型存檔
torch.save(model, 'cnn_model.pth')

# 模型載入
model = torch.load('cnn_model.pth')

# 使用小畫家，繪製 0~9，實際測試看看
from skimage import io
from skimage.transform import resize

no=9
uploaded_file = f'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\myDigits\\{no}.png' # 讀取路徑
image1 = io.imread(uploaded_file, as_gray=True) # 讀取以後轉會成灰階
# io 是 SciKit-Image，一個用於圖像處理的Python庫的子模塊。


data_shape = data.shape # 縮為 (28, 28) 大小的影像
image_resized = resize(image1, data_shape[2:], anti_aliasing=True)  # 從index:2~最後一個切片,反鋸齒  
X1 = image_resized.reshape(*data_shape) #/ 255.0
print(X1[0])
# 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
X1 = 1.0-X1

# 四維: batch, channel, 寬, 高
for i in range(X1[0][0].shape[0]):
    for j in range(X1[0][0].shape[1]):
        print(f'{X1[0][0][i][j]:.4f}', end=' ')
    print()


import matplotlib.pyplot as plt

plt.imshow(X1.reshape(28,28), cmap='gray') # 繪製點陣圖，cmap='gray':灰階

plt.axis('off') # 隱藏刻度 

# 顯示圖形
plt.show() 

# 將非0的數字轉為1，顯示第1張圖片
X2 = X1[0][0].copy() # 創一個X1[0][0]的copy:X2
X2[X2>0.1]=1 # Booling, 符合>0.1的數會=True(1)
print(type(X2), X2[0].shape)
# 將轉換後二維內容顯示出來，隱約可以看出數字為 5

text_image=[]
for i in range(X2.shape[0]):
    text_image.append(''.join(X2[i].astype(int).astype(str))) # 先轉int再轉str
    # ''.join(): 將一行中的所有元素連接成一個字符串。
text_image

X1 = torch.FloatTensor(X1).to(device) # 轉成浮點數tensor

# 預測
predictions = model(X1)
print(np.around(predictions.cpu().detach().numpy(), 2))
print(f'actual/prediction: {no} {np.argmax(predictions.detach().cpu().numpy())}') # 個別機率

model(X1) # X1丟入model


# 讀取影像並轉為單色
for i in range(10):
    uploaded_file = f'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\myDigits\\{i}.png' # 路徑
    image1 = io.imread(uploaded_file, as_gray=True) # 讀進來, 放入image1, 轉成灰階

    # 縮為 (28, 28) 大小的影像
    image_resized = resize(image1, tuple(data_shape)[2:], anti_aliasing=True)    
    #  tuple(data_shape)[2:]: 這是將一個可能是列表或其它可迭代對象的 data_shape 轉換成一個元組，然後使用切片操作從索引 2 開始取得所有後續元素。

    X1 = image_resized.reshape(*data_shape) # reshape(28, 28)
    
    # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
    X1 = 1.0-X1
    
    X1 = torch.FloatTensor(X1).to(device)
    
    # 預測
    predictions = torch.softmax(model(X1), dim=1)
    print(np.around(predictions.cpu().detach().numpy(), 2))
    print(f'actual/prediction: {i} {np.argmax(predictions.detach().cpu().numpy())}')

# 顯示模型的彙總資訊
for name, module in model.named_children():
    print(f'{name}: {module}')

'''
named_children() 方法：這是 torch.nn.Module 的一個方法, 它返回一個迭代器, 遍歷模型的所有直接子模塊。
對於每個子模塊, 它返回一個tuple, 其中第一個元素是子模塊的名稱 (如果在構造函數中指定了名稱)，第二個元素是子模塊對象本身。

for name, module in model.named_children():：這一行啟動一個循環, 遍歷模型的所有直接子模塊。
name 是子模塊的名稱, module 是子模塊的實例。
'''