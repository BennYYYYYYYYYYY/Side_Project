import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# 使用gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu" 

# 資料轉換
# 讀入圖像範圍介於[0, 1]之間，將之轉換為 [-1, 1]
transform = transforms.Compose(  # 以一個序列的方式組合多個轉換操作
    [transforms.ToTensor(), # 將PIL圖像或NumPy ndarray轉換成PyTorch的Tensor。
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     # 進行標準化處理，為了將資料調整到一個更合適的範圍以方便模型學習。
     # 在這裡，每個通道的平均值（mean）和標準差（std）都被設置為0.5
     # 這意味著轉換後的數據的平均值會接近0，標準差接近1。
    ])

# 批量
batch_size = 1000

# 載入資料集，如果出現 BrokenPipeError 錯誤，將 num_workers 改為 0
train_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                        download=True, transform=transform)

# 將組織數據集數據成指定大小的批次，使其可以在訓練過程中使用
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                          shuffle=True, num_workers=2) # num_workers=2：用於數據載入的進程數量。
# 測試集
test_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                       download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# 訓練/測試資料的維度
print(train_ds.data.shape, test_ds.data.shape)

# CIFAR-10數據集中包含的所有類別
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# 圖像顯示函數
def imshow(img):
    img = img * 0.5 + 0.5  # 將圖像數據從[-1, 1]範圍重新調整回[0, 1]範圍。因為之前對圖像進行了標準化
    npimg = img.numpy() # 將圖像Tensor轉換為NumPy陣列
    # 顏色換至最後一維
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # (1, 2, 0)用來指示 np.transpose 函數如何重排 npimg (原始)陣列的軸。
    # PyTorch中的圖像Tensor格式通常是 Channel x Height x Width
    # 而Matplotlib預期的圖像格式是 H x W x C。這行代碼將顏色通道從第一維移至最後一維
    plt.axis('off')
    plt.show()


# 取一筆資料
batch_size_tmp = 8
train_loader_tmp = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_tmp)
# 使用先前定義的數據集 train_ds 和新的批次大小來創建一個數據加載器
dataiter = iter(train_loader_tmp) # 創建的數據加載器轉換為迭代器
images, labels = dataiter.next() # 從迭代器中取出下一批次的數據 images 是圖像的Tensor，labels 是對應的類別標籤
print(images.shape)

# 顯示圖像
plt.figure(figsize=(10,6))
imshow(torchvision.utils.make_grid(images))
# 顯示類別
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_tmp)))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 顏色要放在第1維，3:RGB三顏色
        self.conv1 = nn.Conv2d(3, 6, 5) # 第一個卷積層，輸入通道數為3，輸出通道數為6，使用5x5的卷積核
        self.pool = nn.MaxPool2d(2, 2) # 最大池化層，使用2x2的窗口進行池化操作，步長為2
        self.conv2 = nn.Conv2d(6, 16, 5) # 第二個卷積層，接收前一層的6個輸出通道，輸出通道數增加至16，使用5x5的卷積核
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 第一個全連接層，將展平後的特徵向量（從16個5x5特徵圖）轉換為120個特徵。
        self.fc2 = nn.Linear(120, 84) # 第二個全連接層，將120個特徵減少至84個。
        self.fc3 = nn.Linear(84, 10) # 最後一個全連接層，輸出層，將84個特徵映射到10個類別(CIFAR-10的類別數)

    def forward(self, x): # 向前傳播
        x = self.pool(F.relu(self.conv1(x))) # 通過第一層卷積和激活函數ReLU, 再進行池化
        x = self.pool(F.relu(self.conv2(x))) # 接著數據流經第二個卷積層，同樣經過ReLU激活和池化
        x = torch.flatten(x, 1)  # 數據被展平，以適應全連接層的輸入要求
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) # 經過三個全連接層，每經過一層都應用ReLU激活函數，最後一層直接輸出至10個類別
#         output = F.log_softmax(x, dim=1)
        return x
    
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train() # 訓練mode
    loss_list = []    
    for batch_idx, (data, target) in enumerate(train_loader): # 遍歷訓練資料
        data, target = data.to(device), target.to(device) # 轉移到指定的 device
        
        optimizer.zero_grad() # reset
        output = model(data) # 開始跑data
        loss = criterion(output, target) # 計算預測結果和實際標籤之間的損失
        loss.backward() # backward 計算梯度
        optimizer.step() # update 模型參數
        
        if (batch_idx+1) % 10 == 0: # 每十個批次，記錄當前損失值到 loss_list
            loss_list.append(loss.item())
            batch = (batch_idx+1) * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * (batch_idx+1) / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ' +
                  f'({percentage:.0f} %)  Loss: {loss.item():.6f}')
    return loss_list

def test(model, device, test_loader):
    model.eval() # 評估模式
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 禁用梯度計算
        for data, target in test_loader: # 遍歷 test_loader 中的每一批次數據和標籤
            data, target = data.to(device), target.to(device)
            output = model(data) # 模型對測試數據進行預測
            _, predicted = torch.max(output.data, 1) # 從模型的輸出中選取每個樣本最可能的類別
            # torch.max() 這個函數會返回兩個值：最大值和這些最大值的索引
            # (output.data, 1): 用來從給定的張量中找出某個維度上的最大值, 1表示二維
            correct += (predicted == target).sum().item()

    # 平均損失
    test_loss /= len(test_loader.dataset) 
    # 顯示測試結果
    data_count = len(test_loader.dataset)
    percentage = 100. * correct / data_count 
    print(f'準確率: {correct}/{data_count} ({percentage:.2f}%)')

epochs = 10
lr=0.1

# 建立模型
model = Net().to(device)

# 定義損失函數
# 注意，nn.CrossEntropyLoss是類別，要先建立物件，要加 ()，其他損失函數不需要
'''
使用這個損失函數時，需要首先實例化它，即創建一個對象。
這是因為 nn.CrossEntropyLoss 可能需要一些參數初始化，比如權重或者忽略某些類別的設定。
因此必須使用 nn.CrossEntropyLoss() 來創建一個實例。
'''
criterion = nn.CrossEntropyLoss() # F.nll_loss 

# 設定優化器(optimizer)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) # 使用隨機梯度下降（SGD）作為優化器
'''
momentum 參數是一個增強學習效率和收斂速度的重要技術。
設置 momentum=0.9 意味著在梯度更新過程中，將會考慮之前梯度的累積影響。
具體來說就是利用過去梯度的指數權重平均來調整當前的更新步驟。
這裡的 0.9 表示前一時刻梯度的保留比例。
'''

loss_list = []
for epoch in range(1, epochs + 1): # 透過循環進行多個訓練 epoch，每次調用 train 函數收集損失數據
    loss_list += train(model, device, train_loader, criterion, optimizer, epoch)
    #test(model, device, test_loader)
    optimizer.step()

import matplotlib.pyplot as plt

plt.plot(loss_list, 'r') # show出損失曲線
plt.show()

PATH = './cifar_net.pth' # 儲存model
torch.save(model.state_dict(), PATH)

model = Net() # 創建實例化對象 model，基於定義的 Net 類的結構。
model.load_state_dict(torch.load(PATH))
model.to(device)

test(model, device, test_loader) 

batch_size=8
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size) # 從test_ds數據集中按批次加載數據
dataiter = iter(test_loader)
images, labels = dataiter.next() # 將test_loader轉換成迭代器，這樣可以使用next()函數逐批次讀取數據

# 顯示圖像
plt.figure(figsize=(10,6)) # figure 函數設置圖像顯示的大小 寬度10x長度6
imshow(torchvision.utils.make_grid(images)) 
# torchvision.utils.make_grid 用於將多個圖像組合成一個網格圖

print('真實類別: ', ' '.join(f'{classes[labels[j]]:5s}' 
                         for j in range(batch_size))) # 顯示每個圖像的類別

# 預測
outputs = model(images.to(device))

_, predicted = torch.max(outputs, 1) 
# torch.max函數找出outputs中每個樣本最大值的索引，即模型的預測類別索引

print('預測類別: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size))) # 預測類別

# 初始化各類別的正確數
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# 預測
batch_size=1000 # 增加batchsize 增加整體評估過程
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
# 重新創建一個DataLoader以適應新的batch_size
model.eval()
with torch.no_grad(): # 停止自動梯度計算
    for data, target in test_loader: # 遍歷數據加載器中的所有批次數據
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predictions = torch.max(outputs, 1) # 從輸出中獲得最大值索引，即預測結果(二維=類別)
        # 計算各類別的正確數
        for label, prediction in zip(target, predictions): # 迴圈遍歷批次中的每個標籤和預測
            if label == prediction: # 如果預測正確
                correct_pred[classes[label]] += 1 # 更新正確計數器
            total_pred[classes[label]] += 1 # 無論預測正確與否，都更新該類別的總預測計數器


# 計算各類別的準確率
for classname, correct_count in correct_pred.items():
    # 遍歷correct_pred字典, items(): 在字典上調用，返回一個包含字典中所有鍵值對的迭代器。
    # classname是類別名稱, correct_count是該類別正確預測的數量
    accuracy = 100 * float(correct_count) / total_pred[classname] # 該類別的總預測數量x100 = 百分比 
    print(f'{classname:5s}: {accuracy:.1f} %') # .1表示保留一位小數，f表示浮點數
    #

