'''
超參數調整:
PyTorch推薦使用Ray Tune (pip install ray)
'''
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ray import tune # 管理參數
from ray.tune.schedulers import ASHAScheduler # ASHAScheduler: 基於性能的調度器，可以提前終止性能不佳的試驗。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 檢查GPU

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        # 輸入通道數(in_channels): 1, 代表輸入灰色圖案, RGB的話會是3
        # 輸出通道數(out_channels): 3, 代表卷積層將會產生3個特徵圖(feature maps)的數量
        # Kernel Size: 3, 代表核是3x3的大小 
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        #  (1) self.conv1(x)：數據x首先通過定義好的第一個卷積層conv1。使用卷積核提取輸入數據的特徵
        #  (2) F.max_pool2d(..., 3)：接著對卷積後的特徵圖進行最大池化操作。
        # 池化核大小為3。最大池化是一種降低特徵維度的操作，它通過取局部區域的最大值來簡化信息，
        # 這有助於減少計算量並抑制過擬合。池化核的大小為3意味著每次考慮3x3的區域取最大值。
        #  (3) F.relu(...)：然後對池化後的結果應用ReLU激活函數。
        x = x.view(-1, 192)
        # 調整張量的形狀而不改變其數據內容
        # -1: 表示該維度的大小將自動計算，以使得重塑後的張量與原張量有相同的元素總數。(因為已知特徵圖有192)
        # 192: 重塑後的張量的第二維度的大小，意味著每個新形狀的張量將有192個元素。
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
'''
卷積層輸出的是一個多維張量，通常有形狀(N, C, H, W)，其中N是批量大小（即一次處理的樣本數）、C是通道數、H和W分別是特徵圖的高度和寬度。
全連接層（也稱為密集層或線性層）期望的輸入是一個二維張量，形狀為(N, D)，其中D是特徵維度。
因此，我們需要將卷積層的輸出從四維張量轉換為全連接層能接受的二維形狀。
'''
EPOCH_SIZE = 5

def train(model, optimizer, train_loader):
    model.train()
    for batch_size, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # reset
        output = model(data)
        loss = F.nll_loss(output, target) # loss
        loss.backward() 
        optimizer.step()

def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1) # _ 用於接收最大值本身, predicted 接收 index
            total += target.size(0) # target.size(0): 取他的batch數字, 做更新加總
            correct += (predicted == target).sum().item() # 計算正確的數量(True=1)

    return correct / total

mnist_transforms = transforms.Compose(  # Compose 將多個轉換操作組合成一個鏈式操作
    [transforms.ToTensor(),  # 將PIL圖像或者一個numpy.ndarray轉換成torch.Tensor
     # 不僅將其轉換為張量，還自動將圖像的像素值從[0, 255]範圍縮放到[0.0, 1.0]範圍
     transforms.Normalize((0.1307, ), (0.3081, )) # 標準化參數是書給的
    ])

def train_mnist(config):
    train_loader = DataLoader(
        datasets.MNIST('', train=True, transform=mnist_transforms, download=True),
        # '' 表示將數據集存儲在當前工作目錄下。如果該目錄中沒有找到MNIST數據集，PyTorch會自動從網上下載數據集到這個目錄
        # transform=mnist_transforms應用之前定義的轉換操作，包括轉換成張量和標準化。
        batch_size = 64,
        shuffle = True
    )

    test_loader = DataLoader(
        datasets.MNIST('', train=False, transform=mnist_transforms, download=True),
        batch_size = 64,
        shuffle = True
    )

    model = ConvNet().to(device)  # 建立實體

    optimizer = optim.SGD(model.parameters(),
                        lr=config['lr'], momentum=config['momentum'])
    # config['lr']從一個配置字典中獲取學習率的值，便於調整和優化。
    # 動量（Momentum）是另一個幫助加速SGD在相關方向上並抑制震盪的超參數。
    # config['momentum']從配置字典中獲取動量的值。

    for i in range(10): # 跑10次
        train(model, optimizer, train_loader)
        # 測試
        acc = test(model, test_loader) # test會return準確率

        # 訓練結果(準確率)交給 Ray Tune
        tune.report(mean_accuracy=acc)

        # 每 5 週期存檔一次
        if i % 5 == 0:
            torch.save(model.state_dict(), "model2.pth") # 儲存
            # model.state_dict(): 將每一層與對應的參數（權重和偏置）映射起來

# 參數組合
search_space = { # 定義一個搜索空間字典
    "lr": tune.grid_search([0.01, 0.1, 0.5]), # grid_search: 將會測試列表中的每一個值（0.01、0.1和0.5）
    "momentum": tune.uniform(0.1, 0.9)        # momentum使用uniform(0.1, 0.9)來指定，意味著該值將從0.1到0.9之間均勻隨機選擇。
}

# 執行參數調校
analysis = tune.run(train_mnist, config=search_space, resources_per_trial={'gpu': 1})
# tune.run(): 啟動超參數調整過程。
# config: 指定了超參數的搜索空間。在這被設置為 search_space，定義了要調整的超參數及其搜索範圍。
# 即學習率 lr 設置為在三個值(0.01、0.1、0.5)中進行格子搜索，而動量 momentum 則在 0.1 到 0.9 之間進行均勻抽樣。
# resources_per_trial: 每次試驗(trial)需要的資源。在此{'gpu': 1} 表示每次試驗需要一個 GPU。

for i in analysis.get_all_configs().keys():
    print(analysis.get_all_configs()[i])

import matplotlib.pyplot as plt 

# 取得實驗的參數
config_list = []
for i in analysis.get_all_configs().keys():
    config_list.append(analysis.get_all_configs()[i])
    
# 繪圖
plt.figure(figsize=(12,6))
dfs = analysis.trial_dataframes
for i, d in enumerate(dfs.values()):
    plt.subplot(1,3,i+1)
    plt.title(config_list[i])
    d.mean_accuracy.plot() 
plt.tight_layout()
plt.show()

for i in dfs.keys():
    parameters = i.split("\\")[-1]
    print(f'{parameters}\n', dfs[i][['mean_accuracy', 'time_total_s']])

analysis.results_df

best_trial = analysis.get_best_trial("mean_accuracy", "max", "last")
best_trial.config

logdir = analysis.get_best_logdir("mean_accuracy", mode="max")
state_dict = torch.load(os.path.join(logdir, "model.pth"))

model = ConvNet().to(device)
model.load_state_dict(state_dict)

test_ds = datasets.MNIST('', train=False, download=True, transform=mnist_transforms)

# 建立 DataLoader
test_loader = DataLoader(test_ds, shuffle=False, batch_size=1000)

model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        # 正確筆數
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()

# 顯示測試結果
data_count = len(test_loader.dataset)
percentage = 100.0 * correct / data_count
print(f'準確率: {correct}/{data_count} ({percentage:.0f}%)\n')