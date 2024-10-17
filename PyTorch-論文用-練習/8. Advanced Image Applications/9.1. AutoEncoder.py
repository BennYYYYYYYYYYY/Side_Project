'''
自動編碼器(Autoencoder)

1. 介紹

    自動編碼器是一種用於無監督學習的神經網路結構，其目的是學習資料的高效編碼方式。
    自動編碼器由兩個部分組成：編碼器(Encoder)和解碼器(Decoder)

        1. 編碼器(Encoder): 將輸入資料轉換成低維度的潛在表示(Latent Representation)
        2. 解碼器(Decoder)：將潛在表示轉換回原始資料的高維度表示

    自動編碼器的目標是使重建的資料與原始資料盡可能接近，即最小化重建誤差(Reconstruction Error)
    

2. 運作原理

    1. 編碼器：
        是一個神經網路，其結構可以是多層感知機(MLP)、卷積神經網路(CNN)等。
        其主要目的是將高維度的輸入資料壓縮到一個低維度的潛在空間中。

    2. 解碼器：
        也是一個神經網路，目的是將低維度的潛在表示轉換回高維度的原始資料。

        
3. 訓練過程

    訓練自動編碼器的目標是最小化重建誤差。

    1. 資料準備：收集和標準化訓練資料。
    2. 模型初始化：初始化編碼器和解碼器的權重和偏置。
    3. 前向傳播：將輸入資料通過編碼器和解碼器，計算重建的輸出。
    4. 計算損失：計算重建誤差。
    5. 反向傳播：計算損失對模型參數的梯度。
    6. 參數更新：使用梯度下降法或其他優化算法更新模型參數。
    7. 重複步驟3-6：直到損失函數收斂或達到預定的訓練次數。

    
4. AutoEncode 應用

    1. 資料降維：將高維度資料壓縮到低維度空間，類似於主成分分析(PCA)。
    2. 影像去噪：用於從噪聲影像中重建干淨影像。
    3. 異常檢測：通過比較重建誤差，可以識別出異常樣本。

    
5. 變分自動編碼器(Variational AutoEncode)

    一種生成模型，其目的是學習資料的概率分佈，並可以生成新資料。
    VAE在編碼器和解碼器之間引入了隱變量(Latent Variable)，並使用變分推斷來近似後驗分佈。


    

'''

import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
import random  
import os  # 操作系統相關
import torch  
import torchvision  # PyTorch計算機視覺庫，包含常用的數據集和模型
from torchvision import transforms  # 圖像數據的增強和轉換
from torch.utils.data import DataLoader, random_split  # DataLoader 用於數據加載，random_split 用於分割數據集
from torch import nn  # 構建神經網路的基本模塊，例如層和損失函數。
import torch.nn.functional as F  # 激活函數和損失函數
import torch.optim as optim  
from sklearn.manifold import TSNE  # 用於高維度數據的可視化。t-SNE是一種非線性降維技術。


PATH_DATASETS = ""

# 批次大小
BATCH_SIZE = 256


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# 下載 MNIST 訓練集數據。
train_ds = torchvision.datasets.MNIST(PATH_DATASETS, train=True, download=True)
# 下載 MNIST 測試集數據。download=True: 如果數據集不存在則下載。
test_ds = torchvision.datasets.MNIST(PATH_DATASETS, train=False, download=True)


# 建立一個圖像網格，用於顯示數據集中的圖片
fig, axs = plt.subplots(4, 5, figsize=(8,8))
# 生成一個4x5的子圖，每個子圖大小為8x8英寸

for ax in axs.flatten():
    # 將多維的子圖陣列展平，方便迭代

    # 隨機抽樣一張圖像和其對應的標籤
    img, label = random.choice(train_ds)
    # random.choice 用於從train_ds中隨機選擇一張圖像

    # 顯示圖像
    ax.imshow(np.array(img), cmap='gist_gray')
    # 將圖像轉換為numpy數組並使用灰度顏色顯示

    # 設置圖像標題為其標籤
    ax.set_title('Label: %d' % label)
    # 使用字符串格式化設置標題

    # 移除x軸和y軸的刻度
    ax.set_xticks([])
    ax.set_yticks([])

# 自動調整子圖參數，使圖像不重疊
plt.tight_layout()


# 設置訓練和測試數據集的轉換為張量
train_ds.transform = transforms.ToTensor()
# 將訓練集的轉換設置為ToTensor，將PIL圖像轉換為張量

test_ds.transform = transforms.ToTensor()
# 將測試集的轉換設置為ToTensor，將PIL圖像轉換為張量


# 切割20%的訓練資料作為驗證資料
m = len(train_ds)
# 獲取訓練數據集的總樣本數

train_data, val_data = random_split(train_ds, [int(m - m * 0.2), int(m * 0.2)])
# 使用random_split將訓練數據集分割為訓練集和驗證集，80%作為訓練集，20%作為驗證集


train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
# 創建訓練數據集的DataLoader，每次加載BATCH_SIZE個樣本

valid_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)
# 創建驗證數據集的DataLoader，每次加載BATCH_SIZE個樣本

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
# 創建測試數據集的DataLoader，每次加載BATCH_SIZE個樣本，並隨機打亂數據順序


# 定義編碼器模型
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        # 初始化父類的構造函數

        # 定義卷積層
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # 第1個卷積層：輸入通道1，輸出通道8，卷積核大小3x3，步長2，填充1
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # 第2個卷積層：輸入通道8，輸出通道16，卷積核大小3x3，步長2，填充1
            nn.BatchNorm2d(16),  # 批量歸一化
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.Conv2d(16, 32, 3, stride=2, padding=0),  # 第3個卷積層：輸入通道16，輸出通道32，卷積核大小3x3，步長2，無填充
            nn.ReLU(True)  # 激活函數：ReLU，inplace=True節省內存
        )
        
        self.flatten = nn.Flatten(start_dim=1)  # 將多維輸入展平為一維，從第1維開始展平

        # 定義全連接層
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),  # 全連接層：輸入尺寸3*3*32，輸出尺寸128
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.Linear(128, encoded_space_dim)  # 全連接層：輸入尺寸128，輸出尺寸為編碼空間的維度
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)  # 通過卷積層
        x = self.flatten(x)  # 展平
        x = self.encoder_lin(x)  # 通過全連接層
        return x

    
# 定義解碼器模型
class Decoder(nn.Module):    
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        # 定義全連接層
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),  # 全連接層：輸入尺寸為編碼空間的維度，輸出尺寸128
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.Linear(128, 3 * 3 * 32),  # 全連接層：輸入尺寸128，輸出尺寸3*3*32
            nn.ReLU(True)  # 激活函數：ReLU，inplace=True節省內存
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))  # 將一維輸入恢復為多維

        # 定義轉置卷積層
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),  # 轉置卷積層：輸入通道32，輸出通道16，卷積核大小3x3，步長2，無填充
            nn.BatchNorm2d(16),  # 批量歸一化
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # 轉置卷積層：輸入通道16，輸出通道8，卷積核大小3x3，步長2，填充1，輸出填充1
            nn.BatchNorm2d(8),  # 批量歸一化
            nn.ReLU(True),  # 激活函數：ReLU，inplace=True節省內存
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)  # 轉置卷積層：輸入通道8，輸出通道1，卷積核大小3x3，步長2，填充1，輸出填充1
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)  # 通過全連接層
        x = self.unflatten(x)  # 展平
        x = self.decoder_conv(x)  # 通過轉置卷積層
        x = torch.sigmoid(x)  # 通過Sigmoid激活函數，將輸出限制在[0, 1]範圍內
        return x

    
# 固定隨機亂數種子，以利掌握執行結果
torch.manual_seed(0) # 設置PyTorch的隨機數種子，使結果可重複。

# encoder 輸出個數、decoder 輸入個數
d = 4 # 設置編碼器輸出的維度
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128).to(device)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128).to(device)

loss_fn = torch.nn.MSELoss() # 定義損失函數為均方誤差損失MSELoss
lr= 0.001 # Learning rate

params_to_optimize = [ # 定義需要優化的參數，包括編碼器和解碼器的參數
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr) # 使用Adam優化器來優化參數

def add_noise(inputs,noise_factor=0.3): # 定義函數，為輸入添加噪聲
    noise = inputs+torch.randn_like(inputs)*noise_factor
    noise = torch.clip(noise,0.,1.)
    # noise_factor：噪聲強度
    return noise

def train_epoch_den(encoder, decoder, device, dataloader,  # 定義訓練函數，包括加噪聲、編碼、解碼、計算損失和參數更新。
                    loss_fn, optimizer,noise_factor=0.3):
    # 指定為訓練階段
    encoder.train()
    decoder.train()
    train_loss = []
    # 訓練
    for image_batch, _ in dataloader:
        # 加雜訊
        image_noisy = add_noise(image_batch,noise_factor) # 為圖像添加噪聲。
        image_noisy = image_noisy.to(device)  
        # 編碼
        encoded_data = encoder(image_noisy) # 對噪聲圖像進行編碼
        # 解碼
        decoded_data = decoder(encoded_data) # 對編碼結果進行解碼
        # 計算損失
        loss = loss_fn(decoded_data, image_noisy)
        # 反向傳導
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f'損失：{loss.data}')
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test_epoch_den(encoder, decoder, device, dataloader,  # 定義測試函數，包括加噪聲、編碼、解碼和計算損失。
                   loss_fn,noise_factor=0.3):
    # 指定為評估階段
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        conc_out = [] # 存儲解碼結果
        conc_label = [] # 儲存真實圖像
        for image_batch, _ in dataloader:
            # 加雜訊
            image_noisy = add_noise(image_batch,noise_factor)
            image_noisy = image_noisy.to(device)
            # 編碼
            encoded_data = encoder(image_noisy)
            # 解碼
            decoded_data = decoder(encoded_data)
            # 輸出存入 conc_out 變數
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # 合併
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # 驗證資料的損失
        val_loss = loss_fn(conc_out, conc_label) # 計算驗證損失
    return val_loss.data

# fix 中文亂碼 
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 微軟正黑體，解決中文亂碼問題
plt.rcParams['axes.unicode_minus'] = False # 避免使用負號顯示

def plot_ae_outputs_den(epoch, encoder, decoder, n=5, noise_factor=0.3):
    plt.figure(figsize=(10, 4.5))
    # 創建一個新的圖形，設置圖形大小為10x4.5英寸

    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        # 創建子圖，總共有3行，每行n個子圖，每個子圖位置為 (3, n, i+1)

        img = test_ds[i][0].unsqueeze(0)
        # 從測試集取出第i個圖像，並添加一個維度使其成為[1, 28, 28]

        image_noisy = add_noise(img, noise_factor)
        # 給圖像添加噪聲，噪聲強度由noise_factor控制

        image_noisy = image_noisy.to(device)
        

        encoder.eval()
        decoder.eval()
        # 將編碼器和解碼器設置為評估模式，以防止在前向傳播時應用Dropout等訓練模式下的隨機操作

        with torch.no_grad():
            rec_img = decoder(encoder(image_noisy))
            # 使用編碼器對噪聲圖像進行編碼，然後使用解碼器對編碼結果進行解碼，得到重建圖像

        if epoch == 0:
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            # 顯示原圖像，將張量轉換為numpy數組，並使用灰度顏色顯示
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # 隱藏x軸和y軸的刻度
            if i == n//2:
                ax.set_title('原圖')
                # 如果是第0個epoch，在中間的子圖上設置標題'原圖'

            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
            # 顯示加噪聲的圖像
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # 隱藏x軸和y軸的刻度
            if i == n//2:
                ax.set_title('加雜訊')
                # 如果是第0個epoch，在中間的子圖上設置標題'加雜訊'

        if epoch == 0:
            ax = plt.subplot(3, n, i + 1 + n + n)
        else:
            ax = plt.subplot(1, n, i + 1)
        # 如果是第0個epoch，顯示在第3行；否則顯示在第1行
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        # 顯示重建的圖像
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # 隱藏x軸和y軸的刻度
        if epoch == 0 and i == n//2:
            ax.set_title('重建圖像')
            # 如果是第0個epoch，在中間的子圖上設置標題'重建圖像'

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.3, hspace=0.3)
    # 調整子圖間的間距
    plt.show()
    # 顯示圖形


noise_factor = 0.3
num_epochs = 30
history_da={'train_loss':[],'val_loss':[]}

for epoch in range(num_epochs):
    # 從0到num_epochs-1進行迭代，每次迭代稱為一個epoch

    # 訓練
    train_loss = train_epoch_den(
        encoder=encoder, 
        decoder=decoder, 
        device=device, 
        dataloader=train_loader, 
        loss_fn=loss_fn, 
        optimizer=optim, 
        noise_factor=noise_factor)
    # 訓練模型並返回訓練損失

    # 驗證
    val_loss = test_epoch_den(
        encoder=encoder, 
        decoder=decoder, 
        device=device, 
        dataloader=valid_loader, 
        loss_fn=loss_fn, 
        noise_factor=noise_factor)
    # 驗證模型並返回驗證損失

    # 記錄損失
    history_da['train_loss'].append(train_loss)
    history_da['val_loss'].append(val_loss)
    # 將當前epoch的訓練和驗證損失添加到歷史記錄中

    # 打印當前epoch的訓練和驗證損失
    print(f'EPOCH {epoch + 1}/{num_epochs} \t 訓練損失：{train_loss:.3f}' + 
          f' \t 驗證損失： {val_loss:.3f}')
    # 打印當前epoch的訓練和驗證損失，使用字符串格式化

    # 繪製自動編碼器的輸出
    plot_ae_outputs_den(epoch, encoder, decoder, noise_factor=noise_factor)
    # 繪製自動編碼器的輸出圖像，包括原圖像、加噪聲圖像和重建圖像


test_epoch_den(encoder, decoder, device, test_loader, loss_fn).item()
# 計算測試數據的最終損失，並返回其數值


def plot_reconstructed(decoder, r0=(-5, 10), r1=(-10, 5), n=10):
    plt.figure(figsize=(20, 8.5))
    # 創建一個新的圖形，設置圖形大小為20x8.5英寸

    w = 28
    # 設置每個圖像的寬度為28(MNIST圖像的尺寸)

    img = np.zeros((n*w, n*w))
    # 創建一個大小為 (n*w, n*w) 的零矩陣，將用於存儲重建的圖像

    # 隨機亂數
    for i, y in enumerate(np.linspace(*r1, n)):
        # 生成 n 個從 r1[0] 到 r1[1] 的均勻間隔值，並迭代
        for j, x in enumerate(np.linspace(*r0, n)):
            # 生成 n 個從 r0[0] 到 r0[1] 的均勻間隔值，並迭代
            z = torch.Tensor([[x, y], [x, y]]).reshape(-1, 4).to(device)
            # 創建一個張量 z，其形狀為 (2, 2)，並將其展平成形狀為 (-1, 4) 的張量，然後移動到指定設備

            x_hat = decoder(z)
            # 使用解碼器對 z 進行解碼，得到重建的圖像 x_hat

            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            # 將重建的圖像 x_hat 重新整形為 (28, 28)，並移動到 CPU，轉換為 numpy 數組

            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
            # 將重建的圖像 x_hat 放置到 img 的相應位置

    plt.imshow(img, extent=[*r0, *r1], cmap='gist_gray')
    # 顯示重建的圖像，設置坐標範圍為 r0 和 r1，使用灰度顏色顯示


plot_reconstructed(decoder, r0=(-1, 1), r1=(-1, 1))
# 繪製重建圖像，坐標範圍設置為從 -1 到 1


encoded_samples = []
# 初始化一個空列表，用於存儲編碼樣本

for sample in test_ds:
    img = sample[0].unsqueeze(0).to(device)
    # 取出測試集中的圖像，並添加一個維度使其成為[1, 28, 28]，然後移動到指定設備

    label = sample[1]
    # 取出圖像的標籤

    # Encode image
    encoder.eval()
    # 將編碼器設置為評估模式

    with torch.no_grad():
        encoded_img = encoder(img)
        # 使用編碼器對圖像進行編碼

    # Append to list
    encoded_img = encoded_img.flatten().cpu().numpy()
    # 將編碼後的圖像展平成一維數組，並移動到 CPU，轉換為 numpy 數組

    encoded_sample = {f"變數 {i}": enc for i, enc in enumerate(encoded_img)}
    # 將編碼後的數據存儲到字典中，鍵為變數名稱，值為對應的編碼值

    encoded_sample['label'] = label
    # 將標籤添加到字典中

    encoded_samples.append(encoded_sample)
    # 將字典添加到列表中

encoded_samples = pd.DataFrame(encoded_samples)
# 將列表轉換為 pandas DataFrame


import plotly.express as px
import plotly.graph_objects as go

fig = px.scatter(encoded_samples, x='變數 0', y='變數 1', 
                 color=encoded_samples.label.astype(str), opacity=0.7)
# 使用 Plotly 繪製散點圖，x 軸為變數 0，y 軸為變數 1，顏色根據標籤區分，透明度設置為 0.7

fig_widget = go.FigureWidget(fig)
# 將圖表轉換為 Plotly 圖表小部件，方便在 Jupyter Notebook 中顯示和交互

fig_widget
# 顯示圖表小部件

