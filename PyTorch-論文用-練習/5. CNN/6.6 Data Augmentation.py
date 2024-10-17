'''
Data Augmentation: 數據增強

資料增強是從現有資料以人工方式生成新資料的過程，主要是為了訓練新的機器學習 (ML) 模型。
ML 模型需要大量且多樣化資料集進行初步訓練，但是採購足夠多元性真實資料集可能會因為資料孤島、法規和其他限制而造成挑戰。
資料增強通過對原始資料進行微幅變動來人工增加資料集。生成式人工智慧 (AI) 解決方案現正被用於各行各業，進行高品質和快速的資料增強。
'''

import os  # 處理文件和目錄
import torch 
from torch import nn # 構建神經網絡所需的各種層和功能
from torch.nn import functional as F # 包含各種神經網絡操作的函數
from torch.utils.data import DataLoader, random_split # 提供批量加載數據的迭代器, 隨機拆分數據集
from torchmetrics import Accuracy # 計算模型準確度的度量工具
from torchvision import transforms # 圖像預處理
from torchvision.datasets import MNIST # 廣泛用於手寫數字識別的數據集
import numpy as np 

PATH_DATASETS = "" # 預設路徑
BATCH_SIZE = 1000  # 批量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"


# Data Augmentation函數有很多, 但由於阿拉伯數字有書寫方向, 所以像是水平翻轉就不能用
image_width = 28
train_transforms = transforms.Compose([
    #transforms.ColorJitter(), # 亮度、飽和度、對比資料增補
    # 裁切部分圖像，再調整圖像尺寸
    transforms.RandomResizedCrop(image_width, scale=(0.8, 1.0)), # 隨機選擇圖像中的一部分並裁剪，然後將裁剪得到的圖像調整到指定的尺寸
    # scale=(0.8, 1.0)指定了裁剪大小相對於原始圖像的比例範圍
    transforms.RandomRotation(degrees=(-10, 10)), # 對圖像進行隨機旋轉，角度在-10到10度之間
    #transforms.RandomHorizontalFlip(), # 水平翻轉
    #transforms.RandomAffine(10), # 仿射
    transforms.ToTensor(), # 轉換為PyTorch的Tensor格式
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)) # 對圖像進行標準化，使用平均值0.1307和標準差0.3081
    ])

test_transforms = transforms.Compose([
    transforms.Resize((image_width, image_width)), # 調整圖像的尺寸到28x28像素。這確保了所有測試圖像都有統一的尺寸
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

train_ds = MNIST(PATH_DATASETS, train=True, download=True, 
                 transform=train_transforms)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, # 封裝數據集，提供批次處理、隨機打亂數據及多進程加載等功能。
                                          shuffle=True, num_workers=2) # num_workers=2 表示使用兩個進程來加載數據, 提高加載效率

# 下載測試資料
test_ds = MNIST(PATH_DATASETS, train=False, download=True,  
                 transform=test_transforms)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# 訓練/測試資料的維度
print(train_ds.data.shape, test_ds.data.shape) # 訓練和測試數據的維度，這有助於確認數據的大小和格式

class Net(nn.Module): # 定義名為 Net 的類，繼承自 nn.Module
    def __init__(self): # 初始函數
        super(Net, self).__init__() # 調用父類的構造函數，確保正確地繼承了 nn.Module 的所有屬性和方法
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 二維卷積層，輸入通道數1，輸出通道數32，卷積核大小3x3，步長1
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 二維卷積層，輸入通道數32，輸出通道數64，卷積核大小3x3，步長1
        self.dropout1 = nn.Dropout(0.25) # 隨機丟棄 25% 的神經元
        self.dropout2 = nn.Dropout(0.5) # 隨機丟棄 50% 的神經元
        self.fc1 = nn.Linear(9216, 128) # 全連接層，輸入特徵數為 9216，輸出特徵數為 128
        self.fc2 = nn.Linear(128, 10) # 全連接層，輸入特徵數為 128，輸出特徵數為 10

    def forward(self, x): # 前向傳播路徑
        x = self.conv1(x) 
        x = F.relu(x) # 非線性激活函數 ReLU
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 應用最大池化，使用 2x2 的窗口
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 把二維數據壓平, 批次不用壓
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) # 在最後一層使用 log-softmax 進行分類
        return output
    
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train() # 訓練模式
    loss_list = []    
    for batch_idx, (data, target) in enumerate(train_loader): 
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad() # 清空梯度
        output = model(data) 
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward() 
        optimizer.step() # 更新權重
        
        if (batch_idx+1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx+1) * len(data) # 計算並存儲到目前為止已處理的總數據量
            data_count = len(train_loader.dataset) # 得到訓練數據集的總數據量
            percentage = (100. * (batch_idx+1) / len(train_loader)) # len(train_loader) 表示總批次數, batch_idx+1 除以總批次數再乘以100，得到完成的百分比
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')
    return loss_list # 返回包含每10個批次損失值的列表

def test(model, device, test_loader):
    model.eval() 
    test_loss = 0
    correct = 0
    with torch.no_grad(): 
        for data, target in test_loader: # 循環遍歷測試數據
            if type(data) == tuple: # 如果數據元組形式，則轉換為張量
                data = torch.FloatTensor(data)
            if type(target) == tuple: # label是元組形式，則轉換為張量
                target = torch.Tensor(target)
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1) # 從模型輸出中找到概率最大的類別作為預測類別
            correct += (predicted == target).sum().item() # 累加正確預測的數量

    # 平均損失
    test_loss /= len(test_loader.dataset) 
    # 顯示測試結果
    data_count = len(test_loader.dataset)
    percentage = 100. * correct / data_count 
    print(f'準確率: {correct}/{data_count} ({percentage:.2f}%)')

epochs = 5
lr=1

model = Net().to(device) # 建立模型並將其移到計算設備

criterion = F.nll_loss # nn.CrossEntropyLoss()

# 設定優化器(optimizer)
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

loss_list = []
for epoch in range(1, epochs + 1): # 每次迴圈調用 train 函數, 收集損失值。
    loss_list += train(model, device, train_loader, criterion, optimizer, epoch)
    #test(model, device, test_loader)
    optimizer.step() # 更新參數

import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')
plt.show()

test(model, device, test_loader)

predictions = []
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i][0], test_ds[i][1] # 從 test_ds 數據集中逐一取出前 20 個樣本和其對應標籤
        data = data.reshape(1, *data.shape).to(device) # data.reshape(1, *data.shape) 將單個樣本的數據重塑成批次為1的形式，以符合模型的輸入要求
        output = torch.argmax(model(data), axis=-1) # 從輸出中取得概率最高的類別索引 axis=-1 ,指在最後一維上進行操作,通常是預測類別的維度
        predictions.append(str(output.item()))

# 比對
print('actual    :', test_ds.targets[0:20].numpy()) #  提取前 20 個實際標籤並轉換成 NumPy 陣列形式
print('prediction: ', ' '.join(predictions[0:20])) # 預測結果, 將預測列表轉換為由空格分隔的字符串

import matplotlib.pyplot as plt

def imshow(X):
    # 繪製點陣圖，cmap='gray':灰階
    plt.imshow(X.reshape(28,28), cmap='gray')

    # 隱藏刻度
    plt.axis('off') 

    # 顯示圖形
    plt.show() 

import PIL.Image as Image

data_shape = data.shape

for i in range(10):
    uploaded_file = f'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\myDigits\\{i}.png'
    image1 = Image.open(uploaded_file).convert('L') # 使用 PIL 打開圖像並轉換為灰度模式 ('L' 模式)

    # 縮為 (28, 28) 大小的影像
    image_resized = image1.resize(tuple(data_shape)[2:]) # data_shape[2:] 表示期望的高度和寬度
    X1 = np.array(image_resized).reshape([1]+list(data_shape)[1:])
    # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
    X1 = 1.0-(X1/255)

    # 圖像轉換
    X1 = (X1 - 0.1307) / 0.3081  
    
    # 顯示轉換後的圖像
    # imshow(X1)
    
    X1 = torch.FloatTensor(X1).to(device)
    
    # 預測
    output = model(X1)
    # print(output, '\n')
    _, predicted = torch.max(output.data, 1)
    print(f'actual/prediction: {i} {predicted.item()}')

from skimage import io
from skimage.transform import resize

# 讀取影像並轉為單色
for i in range(10):
    uploaded_file = f'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\myDigits\\{i}.png'
    image1 = io.imread(uploaded_file, as_gray=True) # 使用 skimage 讀取並將圖像轉為灰度格式

    # 縮為 (28, 28) 大小的影像
    image_resized = resize(image1, tuple(data_shape)[2:], anti_aliasing=True) # 重新調整大小(提取高度和寬度這兩個維度)並應用抗鋸齒
    # 批次, 通道數, 高度, 寬度
    X1 = image_resized.reshape([1]+list(data_shape)[1:]) 
    # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
    X1 = 1.0-X1
    
    # 圖像轉換
    X1 = (X1 - 0.1307) / 0.3081  

    # 顯示轉換後的圖像
    # imshow(X1)
    
    X1 = torch.FloatTensor(X1).to(device)
    
    # 預測
    output = model(X1)
    _, predicted = torch.max(output.data, 1)
    print(f'actual/prediction: {i} {predicted.item()}')

class CustomImageDataset(torch.utils.data.Dataset): # 自定義圖像數據集, 用於加載和處理自己的圖像數據, 繼承自 torch.utils.data.Dataset
    def __init__(self, img_dir, transform=None, target_transform=None
                 , to_gray=False, size=28):
        self.img_labels = [file_name for file_name in os.listdir(img_dir)]
        # os.listdir(img_dir): 這是 os 模塊中的一個函數，它返回一個列表，包含指定目錄 img_dir 下的所有文件和目錄的名稱。
        # file_name for file_name in os.listdir(img_dir): 用來迭代 os.listdir(img_dir) 返回的每個文件名，並將每個文件名添加到新列表中
        # 這段語法使用了 Python 的列表推導式 (list comprehension)，這是一種快捷而簡潔的方法來創建列表。
        self.img_dir = img_dir # 存儲圖像文件的文件夾路徑
        self.transform = transform 
        self.target_transform = target_transform
        self.to_gray = to_gray # 布林值，指定是否需要將圖像轉為灰度圖像
        self.size = size # 轉換後的圖像大小

    def __len__(self):
        return len(self.img_labels) # 返回數據集中的圖像總數

    def __getitem__(self, idx):
        # 組合檔案完整路徑
        img_path = os.path.join(self.img_dir, self.img_labels[idx]) # 建立圖像檔案的完整路徑
        # 讀取圖檔
        mode = 'L' if self.to_gray else 'RGB' # 根據 to_gray 參數決定使用 'L' (灰度模式) 或 'RGB'
        image = Image.open(img_path, mode='r').convert(mode) #  打開圖像並轉換為指定模式
        image = Image.fromarray(1.0-(np.array(image)/255)) # 將圖像數據轉換為 NumPy 數組，進行歸一化並反轉顏色

        # print(image.shape)
        # 去除副檔名
        label = int(self.img_labels[idx].split('.')[0]) 
        # 從 self.img_labels 這個列表中取得第 idx 個元素
        # .split('.')[0]: 這部分代碼會將取得的標籤以點 . 作為分隔符號來分割。
        # 選擇 [0] 則是取得分割後列表中的第一個元素
        
        # 轉換
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
ds = CustomImageDataset('C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\myDigits', to_gray=True, transform=test_transforms)
data_loader = torch.utils.data.DataLoader(ds, batch_size=10,shuffle=False) # 創建一個數據加載器，迭代地提供數據。

test(model, device, data_loader) # 調用測試函數, 將模型、設備(CPU或GPU)、和數據加載器作為參數

model.eval()
test_loss = 0
correct = 0
with torch.no_grad(): 
    for data, target in data_loader: # 數據和標籤從數據加載器中迭代獲取
        print(target)
        data, target = data.to(device), target.to(device)
        
        # 預測
        output = model(data) 
        _, predicted = torch.max(output.data, 1) # 得到每個樣本的最大預測概率的索引
        correct += (predicted == target).sum().item() # 計數器更新以記錄正確的預測數量
        print(predicted)

torch.save(model, 'cnn_augmentation_model.pt') # 儲存模型