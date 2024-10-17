# TensorBoard: 視覺化診斷工具, 可以顯示模型結構、訓練過程。 訓練過程使用TensorBoard能夠及時觀看訓練過程
# pip install tensorboard

# 刪除 log 目錄
import os
import shutil
# shutil 模組提供了許多對文件和文件集合進行高級操作的函數，比如複製和刪除。

dirpath = './runs'
if os.path.exists(dirpath) and os.path.isdir(dirpath): # os.path.exists(dirpath) 檢查 dirpath 指定的路徑是否存在
# os.path.isdir(dirpath) 確認這個路徑是否真的指向一個目錄
    shutil.rmtree(dirpath) # shutil.rmtree(dirpath) 被調用來刪除這個目錄及其所有內容。

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 建立transform、trainset、trainloader

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]) # 平均值與標準差=0.5

# datasets
trainset = torchvision.datasets.FashionMNIST('.',
    download=True,
    train=True,
    transform=transform)
 # '.': 存放數據集的路徑。使用 '.' 表示當前目錄。意味著數據集將會被下載並存儲在執行這段代碼的目錄中。

testset = torchvision.datasets.FashionMNIST('.',
    download=True,
    train=False,
    transform=transform)

# dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)
# num_workers:  會創建多個子進程來並行讀取數據。

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# 設定log目錄, 開啟log檔案
from torch.utils.tensorboard import SummaryWriter

# 設定工作紀錄檔目錄
writer = SummaryWriter('runs/fashion_mnist_experiment_1') # SummaryWriter: 記錄訓練過程，方便之後在 TensorBoard 上進行可視化

# 寫入圖片
dataiter = iter(trainloader)
images, labels = next(dataiter) # dataiter.next()：調用迭代器的 next() 方法獲取下一批數據。

# 建立圖像方格
img_grid = torchvision.utils.make_grid(images)
# torchvision.utils.make_grid: 將images中的多個圖像排列成一個網格形式的單一圖像

# 寫入tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)
# 'four_fashion_mnist_images': 標籤名稱
# add_image: 將圖像直接添加到TensorBoard的日誌中，這樣就可以在TensorBoard的界面中查看這些圖像
# writer: 是一個 SummaryWriter 對象的實例，它是用於與 TensorBoard 交互的主要工具
  
# 下載語音資料集
import torchaudio # 處理音頻數據
import os # 讀取目錄內容、創建和刪除目錄、讀取和設置環境變量
import multiprocessing  # 用於創建多進程的模組，它允許利用多核CPU來提高計算密集型任務的執行速度

# 建立目錄
_SAMPLE_DIR = "_sample_data"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no") # 將 _SAMPLE_DIR 和 'yes_no' 這兩部分路徑組合在一起，形成一個完整的路徑
os.makedirs(YESNO_DATASET_PATH, exist_ok=True) # exist_ok=True: 如果目標目錄已經存在，則函數會忽略「目錄已存在」的錯誤
# os.makedirs: 創建目標路徑所指定的目錄。如果路徑中包含了多級尚未存在的目錄，makedirs 會連同中間的所有目錄一起創建

# 讀取資料
def _download_yesno():
    if os.path.exists(os.path.join(YESNO_DATASET_PATH, "waves_yesno.tar.gz")):
    # 先檢查指定路徑下的 'wave_yesno.tarz.gz' 文件是否已經存在。如果這個文件存在，表示YESNO數據集已經被下載，函數將直接返回，不再執行後續的下載操作。
    # root: 用於指定根目錄位置, 「根目錄」（root directory）是指文件系统中的最頂層目錄，其他所有的目錄和文件都包含在根目錄之下。
        return
    torchaudio.datasets.YESNO(root=YESNO_DATASET_PATH, download=True)

    YESNO_DOWNLOAD_PROCESS = multiprocessing.Process(target=_download_yesno) # 創造一個進程
    # multiprocessing.Process: multiprocessing 模塊中用於創建一個新進程的類。一個進程可以被看作是一個獨立的控制流，執行特定的任務。
    # target=_download_yesno: 這個參數指定了新進程啟動後要執行的函數
    
    YESNO_DOWNLOAD_PROCESS.start() # 啟動進程
    # .start(): 這個方法用於啟動進程
    
    YESNO_DOWNLOAD_PROCESS.join()
    # .join(): 這個方法會阻塞（暫停執行）調用它的進程, 直到 YESNO_DOWNLOAD_PROCESS 進程結束執行。


# 語音寫入log
# pip install PySoundFile

'''
waveform 是音頻信號隨時間變化的數位表示，而 sample_rate 則是這個信號被採樣的頻率。
這兩個參數共同定義了一段數位音頻的特性，是進行音頻處理和分析時必須考慮的基礎信息。
'''
from IPython.display import Audio, display 
# 從 IPython.display 模塊中導入了 Audio 和 display 兩個函數。這個模塊是 IPython 核心的一部分，提供了豐富的工具來控制 Jupyter Notebook 中的顯示效果。
# Audio 函數可以接受不同形式的音頻數據輸入，如音頻文件路徑、NumPy 數組等，並將其渲染為可互動的播放器
# display 能夠智能地選擇最佳的顯示方式，例如對於 Audio 對象，它會渲染為音頻播放器

def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()
    # 將 PyTorch 張量轉換成 NumPy 數組，以便後續處理。因為 IPython.display.Audio 需要 NumPy 數組作為輸入

    num_channels, num_frames = waveform.shape # 獲取音頻波形的形狀，即通道數和每個通道的幀數(樣本數)
    if num_channels == 1:
    # 根據通道數來決定如何播放音頻。如果 num_channels 等於 1，說明是單通道（單聲道）音頻
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
    # 如果 num_channels 等於 2，說明是雙通道（立體聲）音頻，此時將兩個通道的波形數據作為一個元組 (waveform[0], waveform[1]) 傳遞給 Audio
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))

# 讀取語音資料集
dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True) # 網上下載

# 讀取3筆資料
for i in [1, 3, 5]: # 選擇數據集中的第2、第4、和第6個樣本
    waveform, sample_rate, label = dataset[i] # 獲取每個音頻樣本的波形數據 (waveform)、採樣率 (sample_rate) 和標籤 (label)
    # 寫入tensorboard
    writer.add_audio('audio_'+str(i), waveform, sample_rate=sample_rate)
    # 播放語音
    play_audio(waveform, sample_rate)


# 使用DataLoader將語音寫入log
# datasets
trainset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH,
                                     download=True)

# dataloader, batch_size必須為1, 否則 next()會出錯, 因為每筆語音長度不同
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,   
                                          shuffle=True)

# 讀取資料
dataiter = iter(trainloader)
# 下一行會出錯, 因為每筆長度不一致, 可能要用transform
waveform, sample_rate, label = next(dataiter)


# 寫入 tensorboard
writer.add_audio('audio', waveform[0], sample_rate=sample_rate.numpy()[0])
# sample_rate=sample_rate.numpy()[0] 將採樣率從 PyTorch 張量轉換為 numpy 數組，並取出第一個元素作為採樣率值。

# 建立模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 第一個 Convolution Layer, 1: 接受單通道圖像, 6: 輸出6個特徵, 5: 使用5x5 Kernel
        self.pool = nn.MaxPool2d(2, 2) # 定義最大池化層: 主要用於減少特徵圖（Feature Map）的空間尺寸，同時保留最重要的特徵信息。
        '''
        最大池化通過應用一個固定大小的窗口（這裡是2x2）掃過前一層的特徵圖，每次移動一定的步長（這裡是2個像素），在每個窗口中選取最大值作為該窗口的輸出。
        這種方式有效地減少了特徵圖的維度，因為每個 2x2 的區域都被壓縮成了一個單一的值。
        窗口首先覆蓋特徵圖左上角的 2x2 區域，從這四個像素中選取最大值作為輸出特徵圖的第一個像素值。
        由於步長為2，窗口接著向右移動兩個像素，繼續進行最大值選取。當窗口到達行末尾時，它會跳到下一行的開始，同樣每次移動兩個像素。
        這一過程在整個原始特徵圖上重複進行，直到覆蓋所有區域。
        '''
        self.conv2 = nn.Conv2d(6, 16, 5) # 第二層 convolution layer, 接收第一層的6個輸入, 並輸出16個特徵, kernal: 5x5
        self.fc1 = nn.Linear(16*4*4, 120) # 第一個全連接層: 輸入16(conc2d輸出特徵圖)*4*4, 輸出120
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 先把x丟入第一層convolution layer, 再用relu激活函數, 再丟到pooling層
        x = self.pool(F.relu(self.conv2(x))) # 再把x丟入第二層convolution layer, relu激活函數, pooling層
        x = x.view(-1, 16*4*4) # 重塑形狀(-1: 自動計算這一維的大小), 2維: 16*4*4
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        # 展平後的數據接著通過兩個全連接層fc1 和 fc2, 每一層之後都應用了relu激活函數
        x = self.fc3(x) # 最後，數據通過第三個全連接層fc3
        return x

net = Net() # 創建model實體

writer.add_graph(net, images)
# add_graph: 需要兩個參數，第一個是模型(net)，第二個是输入到模型的數據(images)
# add_graph會自動提取模型的結構，並記錄下来，以便在TensorBoard中可視化。

# 顯示嵌入向量投影機(Projector)
# 修正 writer.add_embedding 錯誤
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile # 解決在使用 TensorBoard 與 TensorFlow 時可能出現的相容性問題
'''
將 tb.compat.tensorflow_stub.io.gfile 賦值給 tf.io.gfile 的操作: 當需要通過 tf.io.gfile 進行文件操作時，
應該使用 tb.compat.tensorflow_stub.io.gfile 提供的實現。
這樣做的目的通常是為了解決在特定環境下（例如，安裝了TensorBoard但沒有安裝TensorFlow，或TensorFlow和TensorBoard版本不兼容時）可能遇到的問題。
'''

# 隨機抽樣函數
def select_n_random(data, labels, n=100):
    perm = torch.randperm(len(data)) # 使用 torch.randperm 函數生成一個長度為 len(data) 的隨機排列的索引。
    # torch.randperm(n): 生成一個從 0 到 n-1 的整數序列，並將這些整數隨機打亂順序後返回
    return data[perm][:n], labels[perm][:n]
    # data[perm] 利用 perm 中的隨機索引对 data 進行重新排列。
    # data[perm][:n]: 從重排的樣本集合中選取前n個樣本

# 隨機抽樣
images, labels = select_n_random(trainset.data, trainset.targets) # 把trainset的資料丟進剛剛def的 select_n_random函數中並返回images, labels

# 類別名稱
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
'''
模型的輸出一般是一個機率向量，每個元素代表著圖像屬於每個類別的機率。
模型預測的類別索引通常會用這個元組來轉換為人類可讀的類別名稱。
'''

# 轉換類別名稱
class_labels = [classes[lab] for lab in labels]
# for lab in labels: 遍歷 labels，其中每個元素 lab 代表一個樣本的類別標籤（以數字形式表示）
# classes[lab]: 對於每個數字標籤lab，從 classes列表中索引對應的類別名稱。
# 將上述過程的結果收集到一個新的列表中，這個新列表包含了 labels 中每個數字標籤對應的類別名稱，
# 並將這個列表賦值給 class_labels 變量

# 轉為二維向量，以利顯示
features = images.view(-1, 28*28) # 因為資料為28*28像素

# 將 embeddings 寫入 Log 
'''
利用 torch.utils.tensorboard.SummaryWriter 中的 add_embedding 方法來將特徵向量、對應的數據，
以及每個數據點的圖像標籤加入到 TensorBoard 的日誌中。
這樣可以在 TensorBoard 中直觀地查看和分析高維數據在低維空間中的分布情況。
'''
writer.add_embedding(features, metadata=class_labels, # metadata 包含了與 features 中每個數據點相對應的類別標籤
                    label_img=images.unsqueeze(1)) # label_img: 是一個額外的圖像張量，用於提供每個數據點的圖像標籤
# unsqueeze: 用於在指定位置增加一個維度, (1): 意思增加2維的一個維度
writer.flush()
# 使用SummaryWriter紀錄數據到日志文件時，這些數據可能不會立即寫入磁碟。可能先在内存中缓存。
# writer.flush(): 會強制將缓存中的數據寫入到磁碟。可以確保到目前為止使用SummaryWriter紀錄的所有數據都已經被保存
writer.close()
# 完成操作後正確的關閉summarywriter

# 載入 TensorBoard notebook extension，即可在 jupyter notebook 啟動 Tensorboard
# %load_ext tensorboard

# 啟動 Tensorboard
# %tensorboard --logdir=runs