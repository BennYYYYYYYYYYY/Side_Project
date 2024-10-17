# pip install shap

import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim 
from torch.nn import functional as F
import numpy as np
import shap # shap (SHapley Additive exPlanations) 是一個解釋機器學習模型預測的工具庫

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

from torchvision.datasets import MNIST

batch_size = 128
num_epochs = 2

# 下載 MNIST 手寫阿拉伯數字 訓練資料
train_ds = MNIST('.', train=True, download=True, 
                 transform=transforms.ToTensor())

# 下載測試資料
test_ds = MNIST('.', train=False, download=True, 
                 transform=transforms.ToTensor())

# 訓練/測試資料的維度
print(train_ds.data.shape, test_ds.data.shape)

train_loader = torch.utils.data.DataLoader( # 加載 MNIST 數據集
    datasets.MNIST('C:\\Users\\user\\Desktop\\Python\\PyTorch\\CH6\\mnist_data', train=True, download=True, # 數據集存儲在本地 mnist_data 文件夾中；如果該文件夾中沒有數據，則會自動從網上下載
                   transform=transforms.Compose([
                       transforms.ToTensor() # 將圖片轉換為張量
                   ])),
    batch_size=batch_size, shuffle=True) 

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('C:\\Users\\user\\Desktop\\Python\\PyTorch\\CH6\\mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # 調用父類 nn.Module 的初始化方法

        self.conv_layers = nn.Sequential(  # 卷積層
            nn.Conv2d(1, 10, kernel_size=5), # 輸入1 ,輸出10 
            nn.MaxPool2d(2), # 池化, 使用 2x2 的窗口進行池化, 減小特徵圖的空間尺寸
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5), # 第二個卷積層，輸入通道10, 輸出通道20
            nn.Dropout(), # 隨機丟棄一部分神經元(沒有指定數字)
            nn.MaxPool2d(2), # # 池化, 使用 2x2 的窗口進行池化, 減小特徵圖的空間尺寸
            nn.ReLU(), 
        )
        self.fc_layers = nn.Sequential( # 全連接層
            nn.Linear(320, 50), # 輸入320, 輸出50
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10), # 輸入50, 輸出10
            nn.Softmax(dim=1) # 將輸出轉換為概率分佈
        )

    def forward(self, x):
        x = self.conv_layers(x) # 首先通過卷積層
        x = x.view(-1, 320) # 將多維輸入扁平化，為全連接層處理做準備
        # 第二個維度被明確設置為 320，通常基於前面層輸出的特徵數量
        # 第一個維度的 -1 是一個占位符，告訴 PyTorch 自動計算這個維度的大小，以使總元素數量保持不變

        x = self.fc_layers(x) # 通過全連接層序列，得到模型的輸出
        return x

model = Net().to(device)  # model實體 

def train(model, device, train_loader, optimizer, epoch): 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # 遍歷 DataLoader 提供的批次數據, 每個批次包含一組特徵和對應的標籤
        data, target = data.to(device), target.to(device) # 並把 data, target 移到設備
        optimizer.zero_grad() 
        output = model(data)
        loss = F.nll_loss(output.log(), target) # output.log() 表示先對模型的輸出進行 log 轉換，再計算與實際標籤的損失
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # .format() 方法通過 {} 來標記替換的位置，{}內可以放入索引、變量名(如果在 .format() 調用中指定了關鍵字)，或者格式化指示，這些會在執行時替換為實際的變量值
            # :.0f: 整數
            # :.6f: 6小數點float

# 測試函數
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), target).item() 
            # output.log(): 在調用 nll_loss 之前，需要將模型的輸出轉換為對數概率
            pred = output.max(1, keepdim=True)[1]  # keepdim=True: 表示在輸出中保持原有的維度，即使是縮減維度也不會被丟棄
            # [1]: 這個索引用於選取元組中的第二個元素(索引)，即需要的預測類別的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
            # target.view_as(pred): 將 target 張量重新塑形, 使其與 pred 張量的形狀相同
            # eq(...): 是 PyTorch 中的 equal 函數, 用於比較兩個張量中對應元素是否相等, 返回booling
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))      
    # .format() 方法通過 {} 來標記替換的位置，{}內可以放入索引、變量名(如果在 .format() 調用中指定了關鍵字)，或者格式化指示，這些會在執行時替換為實際的變量值
    # {:.4f} 表示打印時將 test_loss 的值格式化為帶有四位小數的浮點數  
    # Accuracy: {}/{} ({:.0f}%): 這裡有三個占位符，前兩個 {} 分別用於顯示變量 correct 和 len(test_loader.dataset) 的值, 代表正確預測的樣本數和測試集的總樣本數。
    # {:.0f}% 用於顯示計算得到的準確率百分比，並將其格式化為沒有小數的整數

    
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # optim.SGD 是 PyTorch 中的隨機梯度下降優化器
# momentum=0.5 使用動量 0.5，這有助於優化器在相關方向上加速

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

batch = next(iter(test_loader)) # 從測試數據加載器中加載一批圖像(先轉成iter再用next取出)
images, _ = batch # _ 表示標籤, 不需要
images = images.to(device)

background = images[:100]  # background 用作解釋背景的數據集，通常選取一部分代表性數據, 這裡選前100個圖像作為背景數據
test_images = images[100:110] # 選取了用於實際解釋的5個圖像 (第101到第105), 這些是想要詳細了解模型預測決策的個別圖像

e = shap.DeepExplainer(model, background) # DeepExplainer 是 SHAP 庫中專為深度學習模型設計的解釋器
# model：要解釋的神經網絡模型
# background：作為參考基線的背景數據集
shap_values = e.shap_values(test_images)   
# 計算指定測試圖像的 SHAP 值。這個函數返回的是一個列表，其中包含對每個類別的解釋，顯示每個輸入特徵對每個類別預測的影響
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values] # 對 shap_values 中每個元素(代表一個圖像的 SHAP 值)進行軸轉換
# np.swapaxes：numpy 中的函数，用於交換數組的兩個軸
# np.swapaxes(s, 1, -1): s是輸入的數組, 1表示原數組中第二個軸, -1表示最後一個軸
# np.swapaxes(np.swapaxes(s, 1, -1), 1, 2): s 的第二軸和最後一軸換, 換好後的第二軸再跟第三軸換

test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2) # 對test_images也進行一樣的軸轉換

# plot the feature attributions
shap.image_plot(shap_numpy, test_numpy)
# 使用 shap.image_plot 函数繪製 SHAP 值。這個函數要求輸入的圖像數據必須具有適當的軸順序, 也就是為何前面要轉換軸順序
# -test_numpy 表示將 test_numpy 中的每個元素取反。這通常用於改變圖像的亮度或對比度，以更好地顯示圖像特徵

