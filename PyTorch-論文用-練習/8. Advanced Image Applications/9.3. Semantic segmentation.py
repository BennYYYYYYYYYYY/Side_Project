'''
1. 語意分割(Semantic segmentation)
    語意分割是一種計算機視覺任務，其目的是將圖像中的【每個像素分類到某一個類別中】。
    與圖像分類不同，因為圖像分類僅對【整張圖像】進行分類，而語意分割則對圖像中的【每一個像素】進行分類。
    例如：有一張包含道路、車輛和行人的街景照片，語意分割的結果會給出一張與原圖相同大小的遮罩圖，標記出每個像素屬於哪一類(例如道路、車輛或行人)

2. U-Net
    U-Net是語意分割中非常著名且常用的神經網路架構。它最早是為了生物醫學影像分割而設計的，但其結構在許多語意分割任務中都表現出色。
    U-Net可以看作是一種特殊的AutoEncoder，只是它在結構上有一些關鍵的不同。

3. U-Net的架構
    U-Net的名字來自於它的形狀，它的結構圖看起來像一個 U 。

        1. 下采樣 (編碼器) 部分：類似AutoEncoder中的編碼器。圖像通過一系列的卷積層逐步變小，提取出圖像中的高階特徵。
        2. 上采樣 (解碼器) 部分：類似AutoEncoder中的解碼器。將縮小的特徵圖逐步放大，直到恢復到原始圖像的大小。
        3. 拼接層 (skip connection)： 
            與普通AutoEncoder不同，U-Net引入了「跳躍連接」技術。
            在解碼過程中，將編碼器中每一層的特徵圖與相對應的解碼器層進行拼接，這樣可以在保留高階特徵的同時，也能恢復更多細節。

            
U-Net的輸入是一張圖像，通過編碼器部分將其轉換成低解析度的特徵圖，然後通過解碼器部分恢復成與輸入相同大小的圖像，同時利用跳躍連接保留了更多的細節和空間信息。
最後，解碼器的輸出是一張與原圖大小相同的【遮罩圖】，其中的每個像素都被分配了一個類別。
 
    【註】
    1. 遮照圖(Mask)：
        在語意分割中，「遮罩圖」是一張與輸入圖像大小相同的圖像，這張圖像不是一張普通的彩色圖像，而是一張用來表示每個像素所屬類別的圖像。

        1. 遮照圖大小：遮罩圖與原圖的大小一致
        2. 像素值：遮罩圖的每個像素值並不表示顏色，而是代表某個特定的類別。
            例如：假設原圖中有一隻貓和一隻狗，遮罩圖會在貓所在的位置用"0"填充，狗所在的位置用"1"填充，其它背景區域可能用"2"來表示
'''

'''
在PyTorch中，Dataset是一個抽象類別 (abstract class)：
    也就是說，它是一個框架，提供了如何去實現一個【自訂的】資料集的結構，但它本身並不直接用來實作具體的功能。
    Dataset提供了一個標準化的方式來封裝和管理數據，使得可以很方便地將數據加載到模型中進行訓練、驗證或測試。
'''

'''
# 範例需要大 GPU 記憶體，需在 Google colab 執行

import os  # 操作系統互動的功能，如檔案與目錄操作。

if not os.path.exists("pytorch_unet"):  # 檢查當前目錄下是否存在名為 "pytorch_unet" 的目錄。
    !git clone https://github.com/usuyama/pytorch-unet.git  # 若目錄不存在，通過 git clone 指令從 GitHub 上複製 pytorch-unet 儲存庫。

%cd pytorch-unet  # %cd 切換當前工作目錄到剛剛克隆的 "pytorch-unet" 目錄下。

!ls  # 列出當前目錄下的所有檔案和資料夾，確認是否成功進入 "pytorch-unet" 目錄。


import torch  

import matplotlib.pyplot as plt  
import numpy as np  
import helper  
import simulation  
from torch.utils.data import Dataset, DataLoader  # 從 PyTorch 導入 Dataset 和 DataLoader ，處理和加載數據集
import torchvision.utils  # 模組提供處理和顯示tensor形式的圖像
from torchvision import transforms, datasets, models  # 從 torchvision 導入 transforms：圖像處理、datasets：數據集、models：訓練模型




if not torch.cuda.is_available():  # 檢查當前系統是否有可用的 CUDA GPU。
    raise Exception("GPU not availalbe. CPU training will be too slow.")  # 如果沒有可用的 GPU，拋出異常，因為在 CPU 上訓練模型會非常慢。

print("device name", torch.cuda.get_device_name(0))  # 如果有可用的 GPU，則打印出 GPU 的名稱。

# 產生3張圖像，寬高各為 192，裡面有6個隨機擺放的圖案。
input_images, target_masks = simulation.generate_random_data(  # 呼叫 simulation 模組中的 generate_random_data 函數。
                                        192, 192, count=3)  # 生成 3 張寬高各為 192 的隨機圖像和對應的目標遮罩(Masks)，這些圖像內包含隨機擺放的圖案。

print("input_images shape and range", input_images.shape, 
      input_images.min(), input_images.max())  # 圖像的形狀和數值範圍，檢查生成的數據是否合理。
print("target_masks shape and range", target_masks.shape, 
      target_masks.min(), target_masks.max())  # 遮罩的形狀和數值範圍，檢查生成的數據是否合理。

# 輸入圖像，改為單色
input_images_rgb = [x.astype(np.uint8) for x in input_images]  # 將輸入圖像的數據類型轉換為 8-bit 整數格式，以確保圖像可以被正常顯示。

# 遮罩(Mask)圖像，使用彩色
target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]  # 使用 helper 模組中的 masks_to_colorimg 函數，將遮罩圖像轉換為彩色圖像，以便更直觀地顯示不同的遮罩區域。

# 顯示圖像：左邊原圖為輸入，右邊遮罩(Mask)圖像為目標
helper.plot_side_by_side([input_images_rgb, target_masks_rgb])  # 使用 helper 模組中的 plot_side_by_side 函數，將輸入圖像和目標遮罩圖像並排顯示，方便比較。


# 自訂資料集：SimDataset 繼承了 Dataset，意味著它遵循 Dataset 的標準結構，並覆蓋(override) __len__ 和 __getitem__ 這兩個方法。

class SimDataset(Dataset):  # 定義一個自訂的資料集類 SimDataset，繼承自 PyTorch 的 Dataset 類。
    def __init__(self, count, transform=None):  # 初始化函數，接收生成的圖像數量 count 和可選的圖像轉換 transform。
        self.input_images, self.target_masks = \
            simulation.generate_random_data(192, 192, count=count)  # 在初始化時，生成指定數量的隨機圖像和對應的目標遮罩圖像。
        self.transform = transform  

    def __len__(self):  # 返回資料集的大小，也就是圖像的數量
        return len(self.input_images)  # 返回 input_images 的長度，即圖像的數量。

    def __getitem__(self, idx):  # 根據索引 idx 返回對應的圖像和mask
        image = self.input_images[idx]  # 根據 idx 獲取對應的輸入圖像
        mask = self.target_masks[idx]  # 根據 idx 獲取對應的 mask
        if self.transform:  
            image = self.transform(image) # 如果有定義轉換函數，則對圖像應用該轉換

        return [image, mask]  # 返回圖像和遮罩的列表。


# 轉換
trans = transforms.Compose([  # 定義圖像轉換組合
  transforms.ToTensor(),  # 將圖像轉換為 tensor 
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 根據 ImageNet 資料集的均值和標準差對圖像進行標準化處理。
])

# 產生訓練及驗證圖像各2000筆
train_set = SimDataset(2000, transform=trans)  # 使用新定義的 SimDataset ，生成 2000 張圖像的訓練集，並應用前面定義的轉換操作。
val_set = SimDataset(200, transform=trans)  # 生成 200 張圖像的驗證集，並應用相同的轉換操作。

image_datasets = {  # 將訓練集和驗證集封裝成字典，以便後續訪問。
  'train': train_set, 'val': val_set
}

batch_size = 25  # 批次大小為 25，在每次迭代時從數據集中讀取 25 張圖像。

dataloaders = {  # 使用 DataLoader 類將資料集分批加載，方便訓練時使用。
  'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),  # 創建訓練集的 DataLoader，num_workers:工作線程數
  'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)  # 驗證集的 DataLoader
}


# 還原轉換
def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))  # tensor -> numpy array， tensor.transpose((原本資料idx, 原本資料idx, 原本資料idx))
    mean = np.array([0.485, 0.456, 0.406])  # 標準化 mean
    std = np.array([0.229, 0.224, 0.225])  # 標準化 std
    inp = std * inp + mean  # 將標準化過的圖像還原為原始圖像的數值範圍。
    inp = np.clip(inp, 0, 1)  # 將圖像中的所有像素值限制在 [0, 1] 的範圍內，以確保圖像顯示正常。
    inp = (inp * 255).astype(np.uint8)  # 將圖像像素值轉換為 8 位整數，準備顯示。

    return inp  # 返回還原後的圖像。

# 取得一批資料測試
inputs, masks = next(iter(dataloaders['train']))  # 從訓練數據加載器中取出一個批次的數據，包括輸入圖像和對應的遮罩。
print(inputs.shape, masks.shape)  # 打印輸入圖像和遮罩的形狀，以確認數據加載正常。
plt.imshow(reverse_transform(inputs[3]))  # 顯示批次中的第四張輸入圖像，並使用 reverse_transform 函數將其還原為可視化格式。

import torch.nn as nn  # 導入 PyTorch 的神經網路模組，提供構建神經網路所需的各種層和函數。
import torchvision.models  # 導入 torchvision 的模型模組，提供多種預訓練模型供使用。

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),  # 定義一個卷積層，處理輸入的特徵圖，並生成新的特徵圖。
        nn.ReLU(inplace=True),  # 定義一個 ReLU 激活函數，對卷積層的輸出進行非線性變換。
    )

class ResNetUNet(nn.Module):  # 定義一個基於 ResNet 和 U-Net 的神經網路模型類，繼承自 PyTorch 的 nn.Module。
    def __init__(self, n_class):
        super().__init__()
        
        # 載入 resnet18 模型
        self.base_model = torchvision.models.resnet18(pretrained=True)  # 載入預訓練的 ResNet18 模型，並將其作為基礎模型。
        self.base_layers = list(self.base_model.children())  # 獲取 ResNet18 模型的各個層級，準備進行修改。

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # 獲取 ResNet18 的前三個層，輸出尺寸為 (N, 64, H/2, W/2)。
        self.layer0_1x1 = convrelu(64, 64, 1, 0)  # 添加一個 1x1 卷積層，用於調整通道數和添加非線性。
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # 獲取 ResNet18 的接下來的兩個層，輸出尺寸為 (N, 64, H/4, W/4)。
        self.layer1_1x1 = convrelu(64, 64, 1, 0)  # 同樣添加一個 1x1 卷積層。
        self.layer2 = self.base_layers[5]  # 獲取 ResNet18 的下一層，輸出尺寸為 (N, 128, H/8, W/8)。
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  # 添加 1x1 卷積層。
        self.layer3 = self.base_layers[6]  # 獲取 ResNet18 的下一層，輸出尺寸為 (N, 256, H/16, W/16)。
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  # 添加 1x1 卷積層。
        self.layer4 = self.base_layers[7]  # 獲取 ResNet18 的最後一層，輸出尺寸為 (N, 512, H/32, W/32)。
        self.layer4_1x1 = convrelu(512, 512, 1, 0)  # 添加 1x1 卷積層。

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 定義上採樣層，使用雙線性插值將特徵圖的尺寸擴大一倍。

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)  # 將 layer4 與 layer3 的輸出進行拼接後進行卷積和激活，特徵圖尺寸為 512 通道。
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)  # 將上一步的輸出與 layer2 的輸出拼接後進行卷積和激活，特徵圖尺寸為 256 通道。
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)  # 將上一步的輸出與 layer1 的輸出拼接後進行卷積和激活，特徵圖尺寸為 256 通道。
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)  # 將上一步的輸出與 layer0 的輸出拼接後進行卷積和激活，特徵圖尺寸為 128 通道。

        self.conv_original_size0 = convrelu(3, 64, 3, 1)  # 對輸入的原始圖像進行初步卷積和激活，生成 64 通道的特徵圖。
        self.conv_original_size1 = convrelu(64, 64, 3, 1)  # 再次進行卷積和激活，保持 64 通道。
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)  # 將最後的上採樣結果與原始特徵圖進行拼接後，進行卷積和激活，回到 64 通道。

        self.conv_last = nn.Conv2d(64, n_class, 1)  # 最後使用 1x1 卷積將特徵圖映射到最終的 n_class 輸出，即每個像素對應的類別。

    def forward(self, input):
        x_original = self.conv_original_size0(input)  # 對原始圖像進行第一次卷積。
        x_original = self.conv_original_size1(x_original)  # 再次卷積。

        layer0 = self.layer0(input)  # 通過 ResNet18 的 layer0(前三層)處理輸入。
        layer1 = self.layer1(layer0)  # 通過 ResNet18 的 layer1(接下來的兩層)處理輸出。
        layer2 = self.layer2(layer1)  # 通過 ResNet18 的 layer2 處理輸出。
        layer3 = self.layer3(layer2)  # 通過 ResNet18 的 layer3 處理輸出。
        layer4 = self.layer4(layer3)  # 通過 ResNet18 的 layer4 處理輸出。

        layer4 = self.layer4_1x1(layer4)  # 將 layer4 的輸出進行 1x1 卷積。
        x = self.upsample(layer4)  # 上採樣 layer4 的輸出。
        layer3 = self.layer3_1x1(layer3)  # 將 layer3 的輸出進行 1x1 卷積。
        x = torch.cat([x, layer3], dim=1)  # 拼接上採樣結果與 layer3 的輸出。
        x = self.conv_up3(x)  # 卷積和激活拼接結果。

        x = self.upsample(x)  # 上採樣輸出。
        layer2 = self.layer2_1x1(layer2)  # 將 layer2 的輸出進行 1x1 卷積。
        x = torch.cat([x, layer2], dim=1)  # 拼接上採樣結果與 layer2 的輸出。
        x = self.conv_up2(x)  # 卷積和激活拼接結果。

        x = self.upsample(x)  # 上採樣輸出。
        layer1 = self.layer1_1x1(layer1)  # 將 layer1 的輸出進行 1x1 卷積。
        x = torch.cat([x, layer1], dim=1)  # 拼接上採樣結果與 layer1 的輸出。
        x = self.conv_up1(x)  # 卷積和激活拼接結果。

        x = self.upsample(x)  # 上採樣輸出。
        layer0 = self.layer0_1x1(layer0)  # 將 layer0 的輸出進行 1x1 卷積。
        x = torch.cat([x, layer0], dim=1)  # 拼接上採樣結果與 layer0 的輸出。
        x = self.conv_up0(x)  # 卷積和激活拼接結果。

        x = self.upsample(x)  # 上採樣輸出。
        x = torch.cat([x, x_original], dim=1)  # 拼接上採樣結果與原始特徵圖。
        x = self.conv_original_size2(x)  # 卷積和激活拼接結果。

        out = self.conv_last(x)  # 最後一層卷積，將特徵圖映射到最終的類別數。
        
        return out  # 返回最終的輸出。



import torch
import torch.nn as nn
import pytorch_unet  # 導入 pytorch_unet 模組，它包含 U-Net 模型相關的功能。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print('device', device)  # 打印使用的裝置。

model = ResNetUNet(6)  # 初始化 ResNetUNet 模型，輸出通道數設為 6，表示最終分類數。
model = model.to(device)  

model  # 在 Jupyter Notebook ，顯示模型的結構。

from torchsummary import summary  # 從 torchsummary 模組中導入 summary 函數，用於顯示模型的摘要資訊。
summary(model, input_size=(3, 224, 224))  # 打印模型摘要，並設定輸入圖像的大小為 (3, 224, 224)。

from collections import defaultdict  # 從 collections 模組中導入 defaultdict，用於儲存模型訓練過程中的度量資料。
import torch.nn.functional as F  # 導入 PyTorch 的函數 API，用於定義損失函數等。
from loss import dice_loss  # 從 loss 模組中導入 dice_loss 函數，用於計算 Dice 損失。

checkpoint_path = "checkpoint.pth"  # 定義檢查點檔案的路徑，用於儲存最佳模型的權重。

# 損失採 binary cross entropy + dice loss
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)  # 計算二元交叉熵損失，適合用於多標籤分類任務。

    pred = torch.sigmoid(pred)  # 將模型輸出經過 sigmoid 函數轉換為概率值。
    dice = dice_loss(pred, target)  # 計算 Dice 損失，評估預測結果與目標之間的重疊程度。

    loss = bce * bce_weight + dice * (1 - bce_weight)  # 結合二元交叉熵損失和 Dice 損失，計算最終的損失值。

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)  # 累積當前批次的二元交叉熵損失。
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)  # 累積當前批次的 Dice 損失。
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)  # 累積當前批次的總損失。

    return loss  # 返回計算出的損失值。

# 計算效能衡量指標
def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append(f"{k}: {(metrics[k] / epoch_samples):4f}")  # 計算每個指標的平均值。

    print(f"{phase}: {', '.join(outputs)}")  # 打印每個階段(訓練或驗證)的指標結果。

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_loss = 1e10  # 初始化最佳損失為一個很大的數字。

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  # 打印當前訓練的 epoch。
        print('-' * 10)

        since = time.time()  # 記錄當前時間，用於計算 epoch 的耗時。

        # 每個 epoch 包含訓練和驗證階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 訓練模式
            else:
                model.eval()   # 驗證模式

            metrics = defaultdict(float)  # 初始化指標字典，用於儲存每個批次的損失。
            epoch_samples = 0  # 記錄每個 epoch 處理的樣本數。

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  
                labels = labels.to(device)  

                optimizer.zero_grad()  # 清空梯度。

                # 前向傳播
                # 如果是訓練階段，則需要追蹤梯度計算過程
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # 通過模型得到輸出。
                    loss = calc_loss(outputs, labels, metrics)  # 計算損失。

                    # 反向傳播 + 優化，只在訓練階段進行
                    if phase == 'train':
                        loss.backward()  # 計算梯度。
                        optimizer.step()  # 更新模型參數。

                # 累積已處理的樣本數
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)  # 打印當前階段的指標結果。
            epoch_loss = metrics['loss'] / epoch_samples  # 計算當前 epoch 的平均損失。

            if phase == 'train':
                scheduler.step()  # 在訓練階段，根據學習率調度器更新學習率。
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])  # 打印當前學習率。

            # 如果在驗證階段並且損失比最佳損失低，則保存模型
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {checkpoint_path}")  # 打印保存模型的資訊。
                best_loss = epoch_loss  # 更新最佳損失。
                torch.save(model.state_dict(), checkpoint_path)  # 保存模型權重。

        time_elapsed = time.time() - since  # 計算一個 epoch 的耗時。
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印耗時。

    print('Best val loss: {:4f}'.format(best_loss))  # 打印最佳驗證損失。

    # 加載最佳模型權重
    model.load_state_dict(torch.load(checkpoint_path))  # 載入訓練期間儲存的最佳模型權重。
    return model  # 返回訓練好的模型。


import torch
import torch.optim as optim  
from torch.optim import lr_scheduler  # 學習率調度器模組，用於動態調整學習率。
import time  # 用於計算訓練過程中的耗時。

num_class = 6  # 定義模型的輸出類別數，這裡設為 6 類。
model = ResNetUNet(num_class).to(device) 

# freeze backbone layers
for l in model.base_layers:  # 使 ResNet backbone 所有層在訓練過程中權重不會更新。
    for param in l.parameters():
        param.requires_grad = False  # 設定 backbone 層的所有參數不需要計算梯度，從而不會更新這些層的權重。

# 使用 Adam 優化器，僅針對 requires_grad = True 的參數進行優化
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 使用 StepLR 調度器，每過 8 個 epoch 將學習率降低 0.1 倍
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

# 訓練模型，設定訓練 10 個 epoch
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=10)

import math  # 用於數學計算。

# 建立新的測試數據集，這裡生成 3 張圖像作為測試數據
test_dataset = SimDataset(3, transform=trans)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)  # 使用 DataLoader 加載測試數據集，批次大小為 3。

# 取出一批測試數據進行測試
inputs, labels = next(iter(test_loader))  # 從測試加載器中取出一個批次的數據。
inputs = inputs.to(device)  # 將輸入數據移動到指定裝置。
labels = labels.to(device)  # 將標籤數據移動到指定裝置。
print('inputs.shape', inputs.shape)  # 打印輸入數據的形狀。
print('labels.shape', labels.shape)  # 打印標籤數據的形狀。

# 預測
model.eval()  # 設置模型為評估模式，以確保在測試過程中不會更新模型權重。
pred = model(inputs)  # 使用模型對輸入數據進行預測。
pred = torch.sigmoid(pred)  # 將預測結果經過 sigmoid 函數轉換為 [0, 1] 之間的概率值。
pred = pred.data.cpu().numpy()  # 將預測結果移動到 CPU 並轉換為 NumPy 陣列。
print('pred.shape', pred.shape)  # 打印預測結果的形狀。

# 還原原圖轉換
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]  # 使用 reverse_transform 函數將原始輸入圖像還原為可視化格式。

# 將標籤轉換為彩色遮罩
target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]  # 使用 helper.masks_to_colorimg 函數將目標遮罩轉換為彩色圖像。

# 將預測結果轉換為彩色遮罩
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]  # 使用 helper.masks_to_colorimg 函數將預測結果轉換為彩色圖像。

# 顯示圖像：左邊:原始圖像，中間:目標遮罩圖像，右邊:模型預測結果
helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
'''





