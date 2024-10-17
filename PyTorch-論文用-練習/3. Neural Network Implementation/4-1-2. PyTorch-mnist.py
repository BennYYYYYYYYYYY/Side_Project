'''
1. 讀取手寫阿拉伯數字的影像，影像中的每一個像素當成一個特徵，每筆資料為寬高(28,28)的點陣圖形
2. 建立神經網路模型，利用梯度下降法(Gradient Descent)求解模型參數，一般稱為權重(Weight)
3. 依照模型去推斷每一個影像是0~9的機率，再以最大機率者為預測的結果
'''
import torch
from pytorch_lightning import LightningModule
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer


# 以PyTOrch Lightning 撰寫
# pip install pytorch-lightning

PATH_DATASETS = '' # 預設資料集路徑是空的
AVAIL_GPUS = min(1, torch.cuda.device_count()) # 若有1個以上gpu就使用gpu，若為0則用cpu
BATCH_SIZE = 256 if AVAIL_GPUS else 64
# BATCH_SIZE(批次大小)：在訓練過程中，一次處理的數據數量。
# 選擇較大的批次大小可以加快訓練速度，但也會增加記憶體的使用量。
# 當使用gpu訓練模型時，通常可以設置更大的批次大小。

# 建立model
class MNISTModel(LightningModule): # 取名(繼承LightningModule)
# Class: object的設計圖, 包含
    # (1) Attribute(屬性): 設計圖的變數
    # (2) Construtor(建構子): 設計圖的函式
    # (3) Method(方法): 設計圖的具體化
    
# Class 特性:
    # (1) 繼承(inheritance): 繼承上一個類
    # (2) 封裝(encapsulation): 只能在class使用, 不能在object使用
    # (3) 多型(polymorphism): 不同的object, 可以互不干擾
     
    def __init__(self): # 初始化, self: 對當前物件的引用
        super().__init__() # 調用父類LightningModule的函數
        self.l1 = torch.nn.Linear(28*28, 10) #創造屬性l1: 輸入28*28, 輸出10的神經網路

    def forward(self, x): # 定義forward方法, 輸入x時調用
        
        # ReLU activation function + Linear網路
        return torch.relu(self.l1(x.veiw(x.size(0), -1)))
        # x.size(0): 返回1維大小, 即batch(批次大小)
        # x.view(): 改變數據大小, -1 代表自動計算符合維度的大小, 此為一維故把28*28

    def training_step(self, batch, batch_nb):
        x, y = batch # 當次數據, x=輸入, y=輸出
        loss = torch.nn.functional.cross_entropy(self(x), y) # 對當前數據的預測self(x)與y真實數據的交叉墒
        return loss 
    
    def configure_optimizers(self): # 使用adam的優化器
        return torch.optim.Adam(self.parameters(), lr=0.02) # self.parameters(): 返回模型中所有可訓練的參數

# 下載MNIST數據
train_ds = MNIST(PATH_DATASETS, train=True, download=True,
                 transform=transforms.ToTensor())
# PATH_DATASETS: 數據集存放的路徑
# train=True: 表示使用的是訓練集
# download=True: 表示如果本地沒有數據集就進行下載
# transform=transforms.ToTensor(): 將圖片轉換為PyTorch張量


# 建立object
mnist_model = MNISTModel() 

# 建立 Dataloader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
# 將數據集 train_ds 按照批次大小 BATCH_SIZE 進行加載

# 模型訓練
trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=3)
# Trainer: 是pytorch-lightning的object, 用於管理訓練過程
# gpus=AVAIL_GPUS: 指定GPU數量

trainer.fit(mnist_model, train_loader)
# trainer.fit: 呼叫訓練循環開始
# mnist_model: 代表model
# train_loader: 代表數據下載器
