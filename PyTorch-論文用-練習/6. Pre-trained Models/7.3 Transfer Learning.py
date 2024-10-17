import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# 訓練資料進行資料增補，驗證資料不需要
data_transforms = { # 這是一個字典, 包含了兩個鍵: 'train' 和 'val'
    'train': transforms.Compose([ # 針對訓練數據的預處理變換
        transforms.RandomResizedCrop(224), # 隨機裁剪圖像的一個區域, 然後調整該區域的大小到 224x224
        transforms.RandomHorizontalFlip(),  # 隨機水平翻轉圖像, 有50%的機率翻轉, 50%的機率保持原樣
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 將圖像的每個通道(RGB)分別標準化, 使用給定的均值和標準差
    ]),
    'val': transforms.Compose([ # 針對驗證數據的預處理變換
        transforms.Resize(256), # 將圖像的較短邊縮放到256像素, 而保持長寬比不變
        transforms.CenterCrop(224), # 從圖像的中心裁剪一個 224x224 的區域
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 使用 ImageFolder 可方便轉換為 dataset
data_dir = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\hymenoptera_data' # 設定數據目錄的路徑, 這個目錄應包含 train 和 val 子目錄, 分別存放訓練和驗證數據
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), # 用字典推導式創建一個字典, 其中的鍵是 train 和 val
                                          data_transforms[x]) # datasets.ImageFolder 用於從目錄中加載影像數據集 , os.path.join(data_dir, x) 將數據目錄與子目錄結合
                  for x in ['train', 'val']} # data_transforms[x] 為每個數據集應用相應的變換
                # image_datasets['train'] 和 image_datasets['val'] 分別對應訓練和驗證數據集
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,# 使用字典推導式創建一個字典, 其中的鍵是 train 和 val
                                             shuffle=True, num_workers=4) # num_workers=4 指定使用 4 個子進程來加載數據
              for x in ['train', 'val']}

# 取得資料筆數
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} # len(image_datasets[x]) 用於獲取訓練和驗證數據集中圖像的數量
'''
這是一種字典推導式語法, 用於創建新的字典。
它會遍歷列表 ['train', 'val'] 中的每個元素 x, 並將 x 作為字典的鍵, len(image_datasets[x]) 作為對應的值。
'''

# 取得類別
'''
torchvision.datasets.ImageFolder 是一個用於圖像數據集的數據加載器。
它假定數據組織在目錄結構中，其中每個子目錄代表一個類別
'''
class_names = image_datasets['train'].classes # image_datasets['train'].classes 取得訓練數據集中的類別名稱
# 這些類別名稱是從數據集中提取的子目錄名稱

dataset_sizes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0)) # 轉換為numpy陣列, 轉置維度, 從 (C, H, W)（通道，高度，寬度）轉換為 (H, W, C)
    mean = np.array([0.485, 0.456, 0.406]) # 標準化的均值和標準差陣列
    std = np.array([0.229, 0.224, 0.225]) 
    inp = std * inp + mean # 將圖像張量進行去標準化處理
    inp = np.clip(inp, 0, 1) # 將圖像數值裁剪到 [0, 1] 範圍內
    plt.axis('off') # 關閉圖像的坐標軸顯示
    plt.imshow(inp) 
    if title is not None: 
        plt.title(title) # 如果提供了標題, 顯示標題
    plt.pause(0.001)  # 暫停一小段時間, 確保圖像繪製更新


# 取得一批資料
inputs, classes = next(iter(dataloaders['train']))
# 使用 iter 和 next 從訓練數據加載器中獲取一個批次的圖像和對應的類別

# 顯示一批資料
out = torchvision.utils.make_grid(inputs)
# torchvision.utils.make_grid 將一批圖像張量組合成一個網格圖像(方便顯示)
imshow(out, title=[class_names[x] for x in classes]) # 調用 imshow 函數顯示網格圖像 out, 並設置標題
# [class_names[x] for x in classes] 這是一個列表推導式, 遍歷 classes 張量中的每個類別標籤 x, 並將其轉換為對應的類別名稱
# class_names[x] 會查找類別標籤 x 對應的類別名稱, 例如，class_names[0] 是 'ants', class_names[1] 是 'bees'

# 同時含訓練/評估
def train_model(model, criterion, optimizer, scheduler, num_epochs=25): # scheduler: 學習率調度器
    since = time.time() # 記錄訓練開始的時間, 以便後續計算訓練所需的總時間

    best_model_wts = copy.deepcopy(model.state_dict()) # 保存模型的初始權重, 作為最佳模型的初始權重
    best_acc = 0.0 # 初始化最佳準確率為 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1)) # 打印當前 epoch 的編號
        print('-' * 10) # 打印分隔線，便於閱讀

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: # 迭代訓練和驗證階段 
            if phase == 'train': 
                model.train()  # 設置模型為訓練模式 (歸一化和dropout)
            else:
                model.eval()   # 設置模型為評估模式 (禁用歸一化和dropout)

            running_loss = 0.0
            running_corrects = 0

            # 逐批訓練或驗證
            for inputs, labels in dataloaders[phase]: # 迭代數據加載器中的每個批次
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad() # 梯度清零

                # 訓練時需要梯度下降
                with torch.set_grad_enabled(phase == 'train'): # 在訓練階段啟用梯度計算
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 取預測的類別(二維最大值的index)
                    loss = criterion(outputs, labels) # 計算損失

                    # 訓練時需要 backward + optimize
                    if phase == 'train': 
                        loss.backward()
                        optimizer.step() # 在訓練階段進行反向傳播並更新參數

                # 統計損失
                running_loss += loss.item() * inputs.size(0) # loss.item() 返回損失的值, 乘以批次大小以得到總損失
                running_corrects += torch.sum(preds == labels.data) # 累加正確預測的數量
            if phase == 'train':
                scheduler.step() # 在訓練階段調整學習率
                # scheduler.step(): 這個方法會更新優化器中的學習率, 根據調度器的規則調整學習率

            epoch_loss = running_loss / dataset_sizes[phase] # 計算 epoch 的平均損失
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # 計算 epoch 的準確率

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc)) # 打印當前階段的損失和準確率

            # 如果在驗證階段且當前準確率優於最佳準確率, 則更新最佳準確率和最佳模型權重
            if phase == 'val' and epoch_acc > best_acc: 
                best_acc = epoch_acc # 更新最佳準確率和最佳模型權重
                best_model_wts = copy.deepcopy(model.state_dict()) # deepcopy 是 copy 模組中的一個方法, 進行深層複製 (deep copy), 即複製一個完全獨立的副本
                '''
                state_dict(): PyTorch提供的一個方法, 用來返回包含模型所有參數(權重和偏置)的字典
                這個字典的Key是參數的名稱, Value是參數的數值
                '''
        print()

    time_elapsed = time.time() - since # 計算訓練所需的總時間
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s') # 打印訓練完成的時間, 格式化為分鐘和秒
    print(f'Best val Acc: {best_acc:4f}') # 打印最佳驗證準確率

    # 載入最佳模型
    model.load_state_dict(best_model_wts)
    return model

def imshow2(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0)) # (C, H, W) 變為 (H, W, C)
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

def visualize_model(model, num_images=6): 
    was_training = model.training # 保存模型當前的訓練狀態, 以便在函數結束時恢復 
    # 它保存了模型當前是處於訓練模式 (training = True) 還是評估模式 (training = False), 在函數結束時, 可以恢復模型到原本的狀態
    model.eval() # 評估模式
    images_so_far = 0 # 初始化已顯示圖像的計數器
    fig = plt.figure() # 建立一個新的圖形窗口

    with torch.no_grad(): 
        for i, (inputs, labels) in enumerate(dataloaders['val']): # 遍歷驗證數據集的 dataloader
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # 獲得每個樣本的預測類別(本人)

            for j in range(inputs.size()[0]): # 遍歷當前 batch 中的所有樣本
                images_so_far += 1 
                plt.subplot(num_images//4+1, 4, images_so_far) # 在圖形窗口中創建子圖, //:整除
                plt.axis('off')
                plt.title(class_names[preds[j]]) # 設置圖像標題為預測類別名稱
                imshow2(inputs.cpu().data[j])

                if images_so_far == num_images: # 如果已顯示的圖像數量達到目標數量
                    model.train(mode=was_training) # 恢復模型的訓練狀態
                    return
        model.train(mode=was_training)
    plt.tight_layout() # 自動調整子圖參數以填充整個圖形區域
    plt.show()

model_ft = models.resnet18(pretrained=True) # 加載預訓練的 ResNet-18 模型

num_ftrs = model_ft.fc.in_features # 獲取最後一層全連接層的輸入特徵數

# 改為自訂辨識層
model_ft.fc = nn.Linear(num_ftrs, 2) # 將最後一層全連接層替換為新的全連接層, 輸出維度為2(兩個分類)

model_ft = model_ft.to(device)

# 定義損失函數
criterion = nn.CrossEntropyLoss() # 使用交叉熵損失函數

# 定義優化器
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# 使用隨機梯度下降(SGD)優化器, 學習率為 0.001, 動量為 0.9

# 每7個執行週期，學習率降 0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft.modules

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25) # 使用 train_model 函數訓練模型

visualize_model(model_ft) # 可視化模型的預測結果



model_conv = torchvision.models.resnet18(pretrained=True) # 加載預訓練的 ResNet-18 模型
for param in model_conv.parameters():
    # 不用重新訓練
    param.requires_grad = False
    # param.requires_grad = False: 凍結這些參數, 使其在訓練過程中不會被更新

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

# 定義損失函數
criterion = nn.CrossEntropyLoss()

# 定義優化器
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

# 每7個執行週期，學習率降 0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)
'''
visualize_model 函數來可視化模型的預測結果。
這個函數展示了訓練或驗證過程中模型的預測圖像及其對應的預測標籤
'''

