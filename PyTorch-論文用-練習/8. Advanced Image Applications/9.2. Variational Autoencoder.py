'''
自編碼器(AutoEncoder)是一種無監督學習的神經網路，主要用於數據的降維和特徵學習。
傳統的自編碼器由兩部分組成：
    編碼器(Encoder)：將輸入數據壓縮成低維度的潛在表示 (latent representation) (高轉低)
    解碼器(Decoder)。根據這個潛在表示重建輸入數據(低轉高)

傳統自編碼器的潛在表示(降維結果)是確定性的，這導致其在生成數據時的能力有限。
    例如： 若希望生成新數據，【傳統自編碼器無法有效地在潛在空間中進行插值或隨機采樣】。為了解決這個問題，變分自編碼器(VAE)應運而生。
        
        【註】：潛在空間
        潛在空間(Latent Space)指的是一個抽象的、低維度的空間，它用來表示數據的核心特徵或本質。在自編碼器或變分自編碼器的架構中，潛在空間通常是由編碼器(Encoder)從高維的輸入數據(如圖片、聲音、文本)中壓縮而來的一組低維度的向量表示。
        這個潛在空間包含了數據中最關鍵的信息，但它的維度要遠小於原始數據的維度。例如，一張 28x28 的灰度圖像有 784 個像素點，但它可能可以壓縮成一個 2 維或 3 維的潛在空間表示。這些低維向量能夠捕捉到圖像中最重要的特徵，忽略掉不必要的細節。
            1. 潛在空間提供了一種壓縮高維數據的方法，使得數據能夠在更小的空間中進行表示。
            2. 潛在空間中，可以通過解碼器(Decoder)將潛在空間中的點轉化為原始數據的重建。
            3. 潛在空間提供了一個可以探索的範圍，我們可以在潛在空間中進行插值，即在兩個點之間找到中間點，並通過解碼器生成一個新的數據點。這使得我們可以在不同類型的數據之間進行平滑過渡。
        潛在空間與原始數據空間的最大區別在於維度的不同。原始空間通常是高維的，充滿了噪聲和冗餘信息。而潛在空間則是一個低維空間，它只保留了數據的最關鍵特徵。
            

        【註】：潛在空間的確定性表示
        傳統自編碼器的編碼器將輸入數據壓縮成一個固定的向量，這個向量是一個確定性的表示。也就是說，對於每一個輸入數據，編碼器都會生成一個唯一的潛在表示。這種確定性的表示使得潛在空間中的向量分佈可能非常不連續，甚至可能出現「離散化」現象。
        例如，假設有兩個輸入數據 X1 與 X2，編碼器將它們分別映射到潛在空間中的兩個點 Z1 與 Z2
        由於這些點是確定性的，潛在空間中可能出現這樣的情況： Z1 與 Z2 雖然在數學上靠得很近，但它們之間的空間可能不代表真實的數據分佈。

        
        【註】：潛在空間中的插值問題
        插值的意思是，在已知兩個點之間找到一個合理的中間點。假設想要在潛在空間中插值，來生成一個介於 Z1 與 Z2 之間的新數據點 Zmid ，這時候傳統自編碼器可能會遇到問題。
        由於潛在空間中可能存在「空洞」或不連續的區域，Zmid 對應的解碼結果可能完全無法代表真實的數據。
        這是因為傳統自編碼器沒有明確控制潛在空間的結構，導致插值出來的點不一定有意義。

        
        【註】：隨機采樣的問題
        隨機采樣指的是在潛在空間中隨機選擇一個點來生成數據。
        傳統自編碼器的潛在空間沒有經過特殊設計或正則化，因此這個空間可能非常不規則。如果我們隨機從潛在空間中選擇一個點，這個點很可能位於“空洞”區域，即不代表任何真實數據的區域。這樣解碼出來的結果也可能是無意義或失真的。

        
VAE 的主要優勢在於它能夠同時進行生成和推斷，並且能夠在潛在空間中進行有效的隨機采樣。
    VAE 將輸入數據映射到一組概率分佈參數(均值和方差)，而不是確定性的潛在向量。
    VAE 的編碼器將輸入數據 x 映射到一個高斯分佈的參數： 均值(𝜇) 和 對數方差(log(σ**2))

    對於每一個輸入數據，編碼器會輸出一個高斯分佈，這個分佈用來表示可能的潛在空間表示。
    這意味著，每次采樣出的新數據都會略有不同，這就引入了隨機性，使得生成的數據具有更多的多樣性。

    
VAE 運作邏輯
    1. 編碼：輸入數據 x 經過編碼器，得到一組均值和方差參數 (𝜇,log(σ**2))
    2. 潛在變量采樣：從高斯分佈中采樣得到潛在變量 z
    3. 解碼：將 z 入解碼器，生成與原始數據相似的數據 x'
    4. 損失計算：計算重建誤差和 【KL散度】，並最小化總損失
     
        【註】：KL 散度
        Kullback-Leibler 散度 (KL 散度)，也稱為相對熵(Relative Entropy)。
        是信息理論中的一個重要概念，用來衡量兩個概率分佈之間的差異程度。衡量的是當使用一個分佈來近似另一個分佈時，會損失多少信息。

'''


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import random 
import os 

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE # 用於高維度數據的可視化。t-SNE是一種非線性降維技術。

PATH_DATASETS = "" # 預設路徑
BATCH_SIZE = 256  # 批量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

train_ds = torchvision.datasets.MNIST(PATH_DATASETS, train=True, download=True)
test_ds  = torchvision.datasets.MNIST(PATH_DATASETS, train=False, download=True)

# 建立一個圖像網格，用於顯示數據集中的圖片
fig, axs = plt.subplots(4, 5, figsize=(8,8))
# 生成一個4x5的子圖，每個子圖大小為8x8英寸

for ax in axs.flatten(): # 將多維的子圖陣列展平，方便迭代
    # 隨機抽樣
    img, label = random.choice(train_ds) # 隨機抽樣一張圖像和其對應的標籤
    ax.imshow(np.array(img), cmap='gist_gray')
    # 將圖像轉換為numpy數組並使用灰度顏色顯示
    ax.set_title('Label: %d' % label)  # 設置圖像標題為其標籤
    ax.set_xticks([])
    ax.set_yticks([]) # 移除x軸和y軸的刻度
plt.tight_layout() # 自動調整子圖參數，使圖像不重疊

# 轉為張量
train_ds.transform = transforms.ToTensor() # 轉換為TENSOR
test_ds.transform = transforms.ToTensor() 

# 切割20%訓練資料作為驗證資料
m=len(train_ds) # 總筆數

# 使用random_split將訓練數據集分割為訓練集和驗證集，80%作為訓練集，20%作為驗證集
train_data, val_data = random_split(train_ds, [int(m-m*0.2), int(m*0.2)])

# 創建訓練數據集的DataLoader，每次加載BATCH_SIZE個樣本
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
# 創建驗證數據集的DataLoader，每次加載BATCH_SIZE個樣本
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)
# 創建測試數據集的DataLoader，每次加載BATCH_SIZE個樣本，並隨機打亂數據順序
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE,shuffle=True)

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        # Convolution
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), # 第1個卷積層：輸入通道1，輸出通道8，卷積核大小3x3，步長2，填充1
            nn.ReLU(True), # 激活函數：ReLU
            nn.Conv2d(8, 16, 3, stride=2, padding=1), # 第2個卷積層：輸入通道8，輸出通道16，卷積核大小3x3，步長2，填充1
            nn.BatchNorm2d(16), # 批量歸一化
            nn.ReLU(True), # 激活函數：ReLU，inplace=True節省內存
            nn.Conv2d(16, 32, 3, stride=2, padding=0),  # 第3個卷積層：輸入通道16，輸出通道32，卷積核大小3x3，步長2，無填充
            nn.ReLU(True)
        )
        
        self.flatten = nn.Flatten(start_dim=1) # 將多維輸入展平為一維，從第1維開始展平

        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128), # 全連接層：輸入尺寸3*3*32，輸出尺寸128
        )
        
        self.encFC1 = nn.Linear(128, encoded_space_dim)  # 編碼器全連接層1：輸出encoded_space_dim維度
        self.encFC2 = nn.Linear(128, encoded_space_dim)  # 編碼器全連接層2：輸出encoded_space_dim維度
        
    def forward(self, x):  # 前向傳播函數，定義輸入數據如何經過編碼器網絡
        x = self.encoder_cnn(x)  # 輸入經過卷積層
        x = self.flatten(x)  # 展平輸出
        x = self.encoder_lin(x)  # 輸入經過全連接層
        mu = self.encFC1(x)  # 計算均值
        logVar = self.encFC2(x)  # 計算對數方差
        return mu, logVar  # 返回均值和對數方差

def resample(mu, logVar):  # 重參數化函數，使用均值和對數方差進行重新採樣
    std = torch.exp(logVar / 2)  # 計算標準差
    eps = torch.randn_like(std)  # 生成與標準差形狀相同的標準正態分佈噪聲
    return mu + std * eps  # 返回重新採樣的結果

class Decoder(nn.Module):  # 定義解碼器模型類
    def __init__(self, encoded_space_dim, fc2_input_dim):  # 初始化函數，設定編碼空間維度和全連接層輸入維度
        super().__init__()

        self.decoder_lin = nn.Sequential(  # 定義一系列全連接層
            nn.Linear(encoded_space_dim, 128),  # 全連接層：輸入encoded_space_dim維度，輸出128維度
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.Linear(128, 3 * 3 * 32),  # 全連接層：輸入128維度，輸出3*3*32維度
            nn.ReLU(True)  # 激活函數：ReLU，inplace=True節省內存
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))  # 將一維數據重構為多維張量

        self.decoder_conv = nn.Sequential(  # 定義一系列反卷積層
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),  # 第1個反卷積層：輸入通道32，輸出通道16，卷積核大小3x3，步長2，無額外填充
            nn.BatchNorm2d(16),  # 批量歸一化，對16個通道進行標準化
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # 第2個反卷積層：輸入通道16，輸出通道8，卷積核大小3x3，步長2，填充1，輸出填充1
            nn.BatchNorm2d(8),  # 批量歸一化，對8個通道進行標準化
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)  # 第3個反卷積層：輸入通道8，輸出通道1，卷積核大小3x3，步長2，填充1，輸出填充1
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)  # 通過全連接層進行線性變換
        x = self.unflatten(x)  # 將一維張量重構為多維張量，準備進行卷積操作
        x = self.decoder_conv(x)  # 通過一系列反卷積層進行上採樣
        x = torch.sigmoid(x)  # 使用Sigmoid激活函數將輸出映射到0到1之間
        return x  # 返回重建後的圖像張量

# 固定隨機亂數種子，以利掌握執行結果
torch.manual_seed(0)  # 設定隨機種子以確保結果的可重現性

# encoder 輸出個數、decoder 輸入個數
d = 4  # 設定編碼器的輸出維度和解碼器的輸入維度
encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128).to(device)  # 初始化並將編碼器模型加載到設備上
decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128).to(device)  # 初始化並將解碼器模型加載到設備上

# KL divergence
def loss_fn(out, imgs, mu, logVar):
    kl_divergence = 0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())  # 計算KL散度，用於衡量輸出的分佈與標準正態分佈之間的差異
    return F.binary_cross_entropy(out, imgs, size_average=False) - kl_divergence  # 計算總損失，包括重構誤差和KL散度

lr = 0.001  # Learning rate 設置學習率

params_to_optimize = [
    {'params': encoder.parameters()},  # 編碼器的參數
    {'params': decoder.parameters()}  # 解碼器的參數
]

optim = torch.optim.Adam(params_to_optimize, lr=lr)  # 使用Adam優化器，並設置學習率

def add_noise(inputs, noise_factor=0.3):
    noise = inputs + torch.randn_like(inputs) * noise_factor  # 在輸入圖像上添加高斯噪聲
    noise = torch.clip(noise, 0., 1.)  # 將噪聲處理後的圖像值限制在0到1之間
    return noise  # 返回加了噪聲的圖像

def train_epoch_den(encoder, decoder, device, dataloader, 
                    loss_fn, optimizer, noise_factor=0.3):
    encoder.train()  # 設置編碼器為訓練模式
    decoder.train()  # 設置解碼器為訓練模式
    train_loss = []  # 用於保存每個batch的損失值

    for image_batch, _ in dataloader:  # 從數據加載器中迭代圖像批次
        image_noisy = add_noise(image_batch, noise_factor)  # 為每個批次圖像添加噪聲
        image_noisy = image_noisy.to(device)  # 將加了噪聲的圖像移到設備上
        mu, logVar = encoder(image_noisy)  # 通過編碼器輸出均值和對數方差
        encoded_data = resample(mu, logVar)  # 根據均值和對數方差進行重新採樣
        decoded_data = decoder(encoded_data)  # 通過解碼器重建圖像
        loss = loss_fn(decoded_data, image_noisy, mu, logVar)  # 計算損失值

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向傳播計算梯度
        optimizer.step()  # 更新參數
        train_loss.append(loss.detach().cpu().numpy())  # 將當前批次的損失值保存

    return np.mean(train_loss)  # 返回當前epoch的平均損失值


def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor=0.3):
    encoder.train()  # 將編碼器設定為訓練模式
    decoder.train()  # 將解碼器設定為訓練模式
    train_loss = []  # 用於保存每個batch的損失值

    for image_batch, _ in dataloader:  # 從數據加載器中迭代獲取圖像批次
        image_noisy = add_noise(image_batch, noise_factor)  # 為圖像批次添加噪聲
        image_noisy = image_noisy.to(device)  # 將加了噪聲的圖像移到設備上
        mu, logVar = encoder(image_noisy)  # 通過編碼器輸出均值和對數方差
        encoded_data = resample(mu, logVar)  # 根據均值和對數方差進行重新採樣
        decoded_data = decoder(encoded_data)  # 通過解碼器重建圖像
        loss = loss_fn(decoded_data, image_noisy, mu, logVar)  # 計算損失值

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向傳播計算梯度
        optimizer.step()  # 更新參數
        train_loss.append(loss.detach().cpu().numpy())  # 保存當前批次的損失值

    return np.mean(train_loss)  # 返回當前epoch的平均損失值


def test_epoch_den(encoder, decoder, device, dataloader, loss_fn, noise_factor=0.3):
    encoder.eval()  # 將編碼器設定為評估模式
    decoder.eval()  # 將解碼器設定為評估模式
    val_loss = 0.0  # 初始化驗證損失為0
    with torch.no_grad():  # 禁用梯度計算，以提高推理速度並節省內存
        conc_out = []  # 用於保存所有批次的解碼輸出
        conc_label = []  # 用於保存所有批次的原始標籤
        for image_batch, _ in dataloader:  # 從數據加載器中迭代獲取圖像批次
            image_noisy = add_noise(image_batch, noise_factor)  # 為圖像批次添加噪聲
            image_noisy = image_noisy.to(device)  # 將加了噪聲的圖像移到設備上
            mu, logVar = encoder(image_noisy)  # 通過編碼器輸出均值和對數方差
            encoded_data = resample(mu, logVar)  # 根據均值和對數方差進行重新採樣
            decoded_data = decoder(encoded_data)  # 通過解碼器重建圖像
            conc_out.append(decoded_data.cpu())  # 將解碼輸出移到CPU並保存
            conc_label.append(image_batch.cpu())  # 將原始圖像移到CPU並保存
            val_loss += loss_fn(decoded_data.cpu(), image_batch.cpu(), mu, logVar)  # 累加每個批次的損失值

        conc_out = torch.cat(conc_out)  # 將所有批次的解碼輸出拼接成一個張量
        conc_label = torch.cat(conc_label)  # 將所有批次的原始標籤拼接成一個張量

    return val_loss.data  # 返回驗證損失值



# 設定Matplotlib使用微軟正黑體，並修正負號顯示問題
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定字體為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 確保負號可以正確顯示

def plot_ae_outputs_den(epoch, encoder, decoder, n=5, noise_factor=0.3):
    plt.figure(figsize=(10, 4.5))  # 創建一個新的圖形，大小為10x4.5英寸

    for i in range(n):  # 迭代n次，生成n個子圖
        ax = plt.subplot(3, n, i + 1)  # 創建子圖，3行n列，當前是第i+1個子圖
        img = test_ds[i][0].unsqueeze(0)  # 從測試數據集中獲取第i個圖像，並添加一個維度(批次大小為1)
        image_noisy = add_noise(img, noise_factor)  # 為圖像添加噪聲
        image_noisy = image_noisy.to(device)  # 將加了噪聲的圖像移到設備上

        encoder.eval()  # 將編碼器設定為評估模式
        decoder.eval()  # 將解碼器設定為評估模式

        with torch.no_grad():  # 禁用梯度計算，以提高推理速度並節省內存
            rec_img = decoder(resample(*encoder(image_noisy)))  # 將噪聲圖像經過編碼器和解碼器生成重建圖像

        if epoch == 0:  # 如果是第0個epoch，繪製原圖和加了噪聲的圖像
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')  # 顯示原圖像
            ax.get_xaxis().set_visible(False)  # 隱藏x軸
            ax.get_yaxis().set_visible(False)  # 隱藏y軸  
            if i == n // 2:  # 在第n//2個子圖上設置標題
                ax.set_title('原圖')

            ax = plt.subplot(3, n, i + 1 + n)  # 創建下一行的子圖
            plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')  # 顯示加了噪聲的圖像
            ax.get_xaxis().set_visible(False)  # 隱藏x軸
            ax.get_yaxis().set_visible(False)  # 隱藏y軸  
            if i == n // 2:  # 在第n//2個子圖上設置標題
                ax.set_title('加雜訊')

        if epoch == 0:  # 如果是第0個epoch
            ax = plt.subplot(3, n, i + 1 + n + n)  # 創建第三行的子圖，顯示重建圖像
        else:
            ax = plt.subplot(1, n, i + 1)  # 否則直接在第一行顯示重建圖像
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  # 顯示重建後的圖像
        ax.get_xaxis().set_visible(False)  # 隱藏x軸
        ax.get_yaxis().set_visible(False)  # 隱藏y軸  
        if epoch == 0 and i == n // 2:  # 在第0個epoch的第n//2個子圖上設置標題
            ax.set_title('重建圖像')

    # 調整子圖之間的間距
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.3, hspace=0.3)  
    plt.show()  # 顯示圖形


noise_factor = 0.3  # 設置噪聲因子，用於向圖像添加噪聲
num_epochs = 50  # 訓練的總epoch數
history_da = {'train_loss': [], 'val_loss': []}  # 初始化字典，用於保存每個epoch的訓練和驗證損失

for epoch in range(num_epochs):  # 迭代進行每個epoch的訓練和驗證
    # 訓練模型
    train_loss = train_epoch_den(
        encoder=encoder, 
        decoder=decoder, 
        device=device, 
        dataloader=train_loader, 
        loss_fn=loss_fn, 
        optimizer=optim,
        noise_factor=noise_factor
    )

    # 驗證模型
    val_loss = test_epoch_den(
        encoder=encoder, 
        decoder=decoder, 
        device=device, 
        dataloader=valid_loader, 
        loss_fn=loss_fn,
        noise_factor=noise_factor
    )

    # 保存每個epoch的訓練和驗證損失
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)

    # 打印當前epoch的訓練和驗證損失
    print(f'EPOCH {epoch + 1}/{num_epochs} \t 訓練損失：{train_loss:.3f}' + 
          f' \t 驗證損失： {val_loss:.3f}')
    
    # 繪製當前epoch的重建圖像、原始圖像、加噪聲圖像
    plot_ae_outputs_den(epoch, encoder, decoder, noise_factor=noise_factor)

# 在測試集上進行最終的驗證，並返回損失值
test_epoch_den(encoder, decoder, device, test_loader, loss_fn).item()

def plot_reconstructed(decoder, r0=(-5, 10), r1=(-10, 5), n=10):
    plt.figure(figsize=(20, 8.5))  # 創建圖形，大小為20x8.5英寸
    w = 28  # 每個小圖像的寬度(像素)
    img = np.zeros((n * w, n * w))  # 初始化大圖像，尺寸為n*w x n*w

    # 遍歷每個位置，生成並繪製解碼器重建的圖像
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y], [x, y]]).reshape(-1, 4).to(device)  # 創建潛在變量張量，並移到設備上
            x_hat = decoder(z)  # 使用解碼器生成重建圖像
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()  # 將重建圖像轉換為numpy數組
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat  # 將重建圖像放置在大圖像的相應位置

    plt.imshow(img, extent=[*r0, *r1], cmap='gist_gray')  # 顯示整體的重建圖像

# 使用訓練好的解碼器進行重建圖像的可視化
plot_reconstructed(decoder, r0=(-1, 1), r1=(-1, 1))


encoded_samples = []  # 初始化一個空列表，用於保存編碼後的樣本

for sample in test_ds:  # 遍歷測試數據集中的每一個樣本
    img = sample[0].unsqueeze(0).to(device)  # 獲取樣本圖像，並添加一個維度(批次大小為1)，將其移動到設備上
    label = sample[1]  # 獲取樣本的標籤
    encoder.eval()  # 將編碼器設定為評估模式
    with torch.no_grad():  # 禁用梯度計算，以提高推理速度並節省內存
        encoded_img = resample(*encoder(img))  # 將圖像通過編碼器進行編碼並重新參數化，獲得編碼後的向量
    encoded_img = encoded_img.flatten().cpu().numpy()  # 將編碼向量展平並轉換為numpy數組，移動到CPU上
    encoded_sample = {f"變數 {i}": enc for i, enc in enumerate(encoded_img)}  # 創建一個字典，將編碼向量的每個元素命名為"變數 i"
    encoded_sample['label'] = label  # 在字典中加入對應的標籤
    encoded_samples.append(encoded_sample)  # 將編碼後的樣本加入列表

encoded_samples = pd.DataFrame(encoded_samples)  # 將列表轉換為pandas DataFrame
encoded_samples  # 顯示編碼後的樣本數據表格

import plotly.express as px  # 導入Plotly Express用於快速繪圖
import plotly.graph_objects as go  # 導入Plotly Graph Objects用於創建更複雜的圖形

# 使用Plotly進行散點圖繪製，顯示變數0和變數1之間的關係，顏色根據標籤分類
fig = px.scatter(encoded_samples, x='變數 0', y='變數 1', 
                 color=encoded_samples.label.astype(str), opacity=0.7)  
fig_widget = go.FigureWidget(fig)  # 將Plotly圖形轉換為FigureWidget以便於進行交互式顯示
fig_widget  # 顯示圖形

# 使用t-SNE進行降維，將編碼向量從高維度空間降到2維空間
tsne = TSNE(n_components=2)  # 初始化t-SNE模型，設置降維後的維度數為2
tsne_results = tsne.fit_transform(encoded_samples.drop(['label'], axis=1))  # 執行t-SNE降維，並丟棄標籤列

# 繪製降維後的散點圖，顏色根據標籤分類
fig = px.scatter(tsne_results, x=0, y=1, color=encoded_samples.label.astype(str),
                 labels={'0': 'tsne-變數1', '1': 'tsne-變數2'})  
fig_widget = go.FigureWidget(fig)  # 將t-SNE圖形轉換為FigureWidget以便於進行交互式顯示
fig_widget  # 顯示圖形


