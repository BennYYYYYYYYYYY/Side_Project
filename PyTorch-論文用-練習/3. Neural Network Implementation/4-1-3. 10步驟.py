import os   # os:處理文件和目錄（也稱為文件夾), 例如使用os模組來讀取、寫入文件、獲取和修改文件的屬性
import torch
from torch import nn # nn:包含了構建神經網絡所需的所有組件，例如各種神經網路層（如全連接層、卷積層等）、激活函數、損失函數
from torch.nn import functional as F  # functional:包含用於神經網絡的構建和操作的函數式, 例如激活(ReLU)、損失(cross entropy)
from torch.utils.data import DataLoader, random_split
# DataLoader: 數據加載器, 可以自動將數據分批次(batch)加載
# random_split: 用於將數據集隨機分割成非重疊的新數據集
from torchmetrics import Accuracy # torchmetrics: 一個專門用於深度學習中計算各種性能指標（如準確率、召回率等）的庫
# Accuracy: 提供計算模型準確率的功能
from torchvision import transforms # torchvision: 一個處理圖像和視頻的PyTorch擴展庫
# transforms: 提供了一系列圖像預處理功能, 如圖像縮放、裁剪、正規化等
from torchvision.datasets import MNIST
# MNIST: 手寫數字識別數據集，包含了大量的手寫數字圖像及其對應的標籤(label)

# 設定參數
PATH_DATASETS = '' # 用於指定MNIST數據集存儲的路徑, 這裡設定為空白
BATCH_SIZE = 1024 # 每次訓練跌代使用1024個數據樣本
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 設定gpu or cpu的使用
'cuda' if torch.cuda.is_available() else 'cpu'

'''
(步驟1)
載入MNIST手寫阿拉伯資料集
'''
# 載入訓練資料
train_ds = MNIST(PATH_DATASETS, train=True, download=True,
                 transform=transforms.ToTensor()) # 用train_ds儲存MNIST的資料集數據
# PATH_DATASETS: 指定數據加載位置
# train=True: MNIST數據分成"訓練集"、"測試集", 這裡指定加載訓練集數據
# download=True: 這個參數控制當數據集在指定的存儲路徑上不存在時，是否自動從互聯網下載數據集。
# transform=transforms.ToTensor(): 加載數據時應用於數據的轉換操作, 把數據轉成tensor(圖片的話為 0~255)

'''
補充說明: download=True
 (1) 當數據集在指定的存儲路徑上不存在時： 
意味著系統會首先檢查你提供的存儲路徑（PATH_DATASETS指向的文件夾）來確定該數據集是否已經被下載並存儲在那裡。
如果系統在這個路徑上找到了數據集的相關文件，就會直接使用這些文件，不會再進行下載。
 (2) 是否自動從互聯網下載數據集：
如果在指定的存儲路徑上沒有找到數據集，download=True這個參數會告訴系統自動從網上找到這個數據集的來源，
並將其下載到PATH_DATASETS指定的文件夾。這個過程是自動進行的，無需手動介入。
'''

# 下載測試資料
test_ds = MNIST(PATH_DATASETS, train=False, download=True, # 跟train_ds很像, 只是train=False: 代表下載測試集資料
                transform=transforms.ToTensor())

# 訓練/測試集的維度
print(train_ds.data.shape, test_ds.data.shape) 
# torchvision.datasets中，加載的數據集像MNIST這樣的類包含了許多屬性和方法
# .data屬性專門用於存儲數據集中的原始圖像數據(x), 可以保證獲取的是未經任何預處理或轉換的數據
# .targets屬性專門儲存標準答案(y)
'''
補充說明: train_ds.data.shape
數據集對象如MNIST並不直接提供.shape屬性來查詢整個數據集的維度或形狀。這是因為數據集對象被設計為一個可迭代的容器，
裡面包含了多個元素（在這個場合是圖像和標籤的對），而不是一個單一的數據結構（如張量或NumPy數組）。
'''


'''
(步驟2)
對資料集進行探索與分析(Exploratory Data Analysis, EDA)
'''

# 首先觀察訓練資料的目標值/標籤(y), 即影像的真實結果 (把訓練集資料的目標(target)調出前10個來 )
train_ds.targets[:10] 

# 然後把訓練集資料調出來(第一個x)
print(train_ds.data[0]) 
'''
補充說明: print(train_ds.data[0])

結果: 每筆像素值在0~255之間, 0=白色, 255最黑的黑色
'''

# 為了看清楚手寫的數字, 把非0(不是白色的)的數字轉成1, 變成黑白兩色圖片
data = train_ds.data[0].clone() # 複製 train_ds.data第一個資料, 並放入data中
data[data>0]=1  # 把data中>0的數值都轉成1
data = data.numpy() # 把tensor轉成numpy陣列

# 把轉換後的二維內容顯示出來，隱約可以看到數字'5'
text_image=[] # 創造一個空的列表text_image
for i in range(data.shape[0]): # 遍歷 data.shape[0]: 即data中的第行(第一維)
    # 若以圖像的二值化表示，data.shape[0]就是圖像的高度（有多少像素行）
    text_image.append(''.join(data[i].astype(str))) 
    # ''.join(): 將多個字符串連接成一個單一的字符串, ''表示連接時不加任何分隔符
    # data[i]: 把每一行跌代出來並轉換成str再整個串起來
    # astype(str): 是numpy數組的一個方法，用於將數組的DataType轉成指定的類型，這裡是將數字轉換成string
text_image


# 將非0的數字轉為1，顯示第2張圖片
data = train_ds.data[1].clone()  
data[data>0]=1
data = data.numpy()

# 將轉換後二維內容顯示出來，隱約可以看出數字為 5
text_image=[]
for i in range(data.shape[0]):
    text_image.append(''.join(data[i].astype(str)))
text_image


# 顯示第1張圖片圖像
import matplotlib.pyplot as plt

# 第一筆資料
X = train_ds.data[0]

# 繪製點陣圖，cmap='gray':灰階
plt.imshow(X.reshape(28,28), cmap='gray')
# imshow: 用於顯示圖像的函數。這個函數接受一個"圖像數據"作為輸入，並將其顯示在一個圖形窗口中。
# imshow: 函數接受一個二維數組輸入, 適用於需要直接展示圖像數據的場合
# X.reshape(28, 28): 把向量轉成2維(28,28), 這樣做是為了確保圖像以正確的形狀顯示
# camp='gray': colormap的縮寫, 它指定了圖像的顏色映射方案, 這裡指定為"灰階"

# 隱藏軸
plt.axis('off') 
# 默認為會開啟軸訊息

# 顯示圖
plt.show()


'''
(步驟3)
使用PyTorch進行特徵縮放時, Dataloader載入資料時才會進行
要設定縮放可在下載MNIST指令內設定
'''
transform = transforms.Compose([
# 當有多個需要按特定順序應用到每個輸入樣本上的轉換操作時，transforms.Compose提供了一個簡潔的方式來實現這一點。
# 只需將這些轉換作為列表傳遞給Compose，這個新轉換會按列表中的順序依次應用每個轉換。
    transforms.ToTensor(), # 轉換成tensor, 並自動進行特徵縮放(從0~255, 轉換成0.0~1.0)
    transforms.Normalize((0.1307,), (0.3081)) # 將數據標準化: (input-mean)/std, 這裡的mean:0.1307, std:0.3081
])

train_ds = MNIST(PATH_DATASETS, train=True, download=True,
                 transform=transform) # 下載MNIST數據集
# transform=transforms: 對每個MNIST數據集資料進行transform預處理


'''
(步驟4)
資料分割為訓練以及測試集資料, 但此處不需要進行, 因為載入MNIST資料時就已經切好了
'''


'''
(步驟5)
建立模型結構: Pytorch提供兩類模型
 (1) 順序型模型(Sequential Model): torch.nn.Sequential
 (2) Functional API模型: torch.nn.functional
'''
# 建立model
model = torch.nn.Sequential(
    torch.nn.Flatten(), # 全部壓成1維
    torch.nn.Linear(28*28, 256), # 輸入784個特徵, 並輸出256個特徵
    nn.Dropout(0.2), # 丟棄20%數據, 避免overfitting的問題
    torch.nn.Linear(256, 10) # 輸入256個特徵, 並輸出10個(由於要預測的數字為0~9)
    ).to(device) # device: gpu與cpu不可混用
'''
補充: 使用 nn.CrossEntropyLoss()時, 不須要經過softamx層, PyTorch已內涵softmax處理
即 torch.nn.Softmax(dim=1), (dimension=1, 即計算2維)
'''


'''
(步驟6)
結合訓練資料即模型, 進行模型訓練
'''
epochs = 5
lr = 0.1

# 建立Dataloader
train_loader = DataLoader(train_ds, batch_size=600) # 每一批次取600個數據與他的標籤
# DataLoader: PyTorch中用於加載數據的一個工具，它可以從一個給定的數據集中批量地提取數據，並提供多種數據迭代的方式

# 設定優化器(optimizer)
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)  # model.parameters(): 待優化的參數集合(返回模型中所有需要被優化的參數)
criterion = nn.CrossEntropyLoss() # 交叉商損失函數

model.train() # 開啟訓練模式
'''
補充: model.train()訓練模式、model.eval()評估模式
 (1) model.train(): 意味著某些層（如Dropout）會按照訓練時的行為來運作, 例如Dropout層會開始隨機丟棄神經元
 (2) model.eval(): 意味著Dropout層被禁用
 意思是在模型做預測或者進行測試的時候，我們讓模型的表現（它是怎麼處理數據、怎麼給出結果的）和在訓練時保持一樣。
'''
loss_list = []
for epoch in range(1, epochs+1): # 跑epochs次迴圈(5次): (開始(包含), 結束(不包含))
    for batch_idx, (data, target) in enumerate(train_loader): # 設置一個循環來遍歷訓練數據集，數據集被DataLoader分割成了多個批次（batch）
    # enumerate(列舉): 迭代一個序列的同時跟蹤元素的索引(index)
    # 這裡會讓enumerate()函數輸出: train_loader的索引(即batch_idx) 以及內容(即data與target)
        
        data, target = data.to(device), target.to(device) # 將data與target移動到device(GPU or CPU)
    #  if batch_idx == 0 and epoch == 1: print(data[0])

        optimizer.zero_grad() # reset
        output = model(data) # 向前傳導: 把data(即x)丟入定義好的模型中(model), 並產出預測output
        loss = criterion(output, target) # 使用預測值與目標值, 進行損失值的計算
        loss.backward() # 反向傳導
        optimizer.step() # 更新參數

        if batch_idx % 10 == 0: # 如果跌代的批次index為10的整數時: 
            loss_list.append(loss.item()) # append損失值進loss_list[]中, loss.item: 轉為常數
            batch = batch_idx * len(data) # 計算到目前為止處理過的數據總量, index*數據量
            data_count = len(train_loader.dataset) # 算train_loader的數據總數(獲得數據集的資料總數)
            percentage = (100. * batch_idx/len(train_loader)) # 當前批次/總數據批次*100%
            # 用100.的原因是要用浮點數
            print(f'Epoch{epoch}:[{batch:5d}/{data_count}]({percentage:.0f} %)' +
                  f' Loss:{loss.item():.6f}')
            # :5d(decimal十進位代表為整數): 整數格式化為至少5位的寬度。如果整數本身的位數少於5，則其前面會補上空格，以確保總寬度為5
            # :.0f(fixed point固定小數點代表為浮點數): 小數點後保留0位, 即四捨五入到整數位
            # :.6f: 四捨五入到小數第6位

import matplotlib.pyplot as plt 
plt.plot(loss_list, 'r')
plt.show()

'''
(步驟7)
評分(Score Model), 輸入測試資料, 計算出損失Loss與準確度Accuracy
''' 
# 建立DataLoader
test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE) # 不對數據進行隨機洗牌shuffle=False

model.eval() # 評估模式
test_loss = 0
correct = 0
with torch.no_grad(): # 不進行梯度下降(因為訓練時已經練出最佳的參數)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data) 

        # 加總
        test_loss += criterion(output, target).item()

        # 預測
        pred = output.argmax(dim=1, keepdim=True)
        # argmax(): 找到最大值的索引, dimensin=1(找二維數據, 即特徵)
        # keepsim參數: 決定tensor操作後的輸出是否保留原始的維度
        # keepdim=True: 出張量仍然保持與原始張量相同的維度數但是，在被操作的維度上，其大小將變為1。

        # 正確筆數
        correct += pred.eq(target.view_as(pred)).sum().item()
        # view_as(): 把tensor形狀調整成另一個指定tensor的形狀
        # eq(): 比較操作, 比較裡面的兩個值, 若一樣返回1, 不一樣則返回0, 這裡是比較pred與target
        # 最後把比較後的結果sum在一起, 最後轉換成python數字(就能知道共有幾筆資料正確)

# 平均損失
test_loss /= len(test_loader.dataset) # 損失值/test_loader的資料總數量

# 顯示測試結果
batch = batch_idx * len(data) # 計算數據總量, batch_idx: 數據batch index, data: 每批次的數據
data_count = len(test_loader.dataset) # 測試數據集的總樣本數量
percentage = 100. * correct/data_count # 正確/總量
print(f'平均損失:{test_loss:.4f}, 準確率:{correct}/{data_count}'+
      f'({percentage:.0f}%)\n')

# 實際比對資料的前20筆
predictions = []
with torch.no_grad(): 
    for i in range(20): # 預測前20個
        data, target = test_ds[i][0], test_ds[i][1]  # 從test_ds中提取第i筆資料的第1個(即輸入值/特徵), 與第2個(標籤/目標值)
        data = data.reshape(1, *data.shape).to(device) # 在數據最前面新增一個維度, 
        # *data.shape: 把data.shape 解包並傳給reshape, 變成data.reshape(1, 28, 28), 即新增一個維度 
        output = torch.argmax(model(data), axis=1) # 找出每行最大值的index
        predictions.append(str(output.item())) # append到predictions list (str)

# 比對
print('actual: ', test_ds.targets[0:20].numpy()) # 前20個並轉numpy數列
print('prediction: ', ''.join(predictions[0:20])) # 將列表中的所有字符串元素連接成一個長字符串


# 顯示第九筆資料的機率
import numpy as np
i = 8
data = test_ds[i][0] # 取出第i個的特徵
data = data.reshape(1, *data.shape).to(device) # reshape成(1, 28, 28) (把data.shape解包)
predictions = torch.softmax(model(data), dim=1) # 把data結果的第二維度輸出(label)換成機率表示(sum=1)
print(f'0~9預測機率: {np.around(predictions.cpu().detach().numpy(), 2)}') # detach()：將tensor從計算圖中分離, 避免影響梯度自動運算
print(f'0~9預測機率: {np.argmax(predictions.cpu().detach().numpy(), axis=-1)}') # axis=-1: 在最後一個維度中進行(類別/目標/標籤)
'''
補充: detach()
只是要查看模型的輸出結果，或將模型的輸出結果轉成NumPy數組以便進行後續的處理時，不需要對這些結果進行梯度更新。
如果這時不使用.detach()，即使不進行任何梯度計算，tensor背後的計算歷史仍然會被保留，意味著即使是不需要進行反向傳播的情況，
記憶體也會因為保存了不必要的計算歷史而被佔用。.detach()可以將tensor從計算歷史中分離出來，創建一個新的tensor，不會保存任何計算歷史。
'''

# 顯示第九筆圖像
x2 = test_ds[i][0]
plt.imshow(x2.reshape(28,28), cmap='gray')
# imshow: 常將2維數據直接轉化成圖像
# colormap: 參數決定了數據的數值如何轉換成顏色, 它是用於二維數據或更高維數據的顯示, 控制不同數值對應到的顏色
plt.axis('off') # 關閉axis
plt.show()


'''
(步驟8)
效能評估: 暫不進行, 之後可調整超參數(Hyperparameter)及model結構, 尋找更佳的模型與參數
'''


'''
(步驟9)
模型部屬: 將最佳模型存檔, 再開發使用者介面或提供API, 連同檔案一併部屬(Deployment)到正式環境(Productive Enviornment)
Productive Enviornment: 部署並運行軟體應用的最終階段，這個環境中的應用對終端用戶是可見可用的。相比之下，開發環境、測試環境等則用於軟體的開發和測試階段，不對外提供服務。
'''
# # 模型結構與權重一起存檔
torch.save(model, 'model.pt') 
# model: 儲存的模型, 包含架構與所有權重+偏差
# 'model.pt': 儲存的路徑與名稱

# 模型載入
model = torch.load('model.pt') # 副檔名通常用.pt
# torch.load('model.pt'): 從路徑中載入之前儲存的PyTorch模型, 並賦予給model


# (只儲存參數的) 權重+偏差儲存(不儲存架構)
torch.save(model.state_dict(), 'model.pth')
# .state_dict(): 返回模型參數的字典

# 載入權重
model.load_state_dict(torch.load('model.pth')) # 載入時需要先有模型架構的實例(即這裡的model)
# .load_state_dice(): 接受一個參數字典作為輸入並將這些參數（權重和偏差）賦值給模型的對應參數


'''
(步驟10)
系統上線,提供給新資料做預測(就不用MNIST的了), 這裡使用myDigits目錄的圖
'''
# 從圖檔讀入影像後要反轉顏色, 顏色0為白色, 與RGB不同,RGB的0是黑色
# 使用skimage套件讀取圖檔, 像素會自動縮放至[0,1]之間, 不須自己做轉換
from skimage import io
from skimage.transform import resize
# skimage: 是Python的一個圖像處理庫，提供了許多用於圖像處理和分析的工具
# io: 包含讀取、保存和顯示圖像的函數
# transform: 提供許多圖像變換和操作的函數，其中resize函數用於改變圖像的尺寸

# 讀取並轉換成灰色
for i in range(10):
    uploaded_file = f'C:\\Users\\user\\Desktop\\Python\\pytorch data\\myDigits\\{i}.png' # 動態生成文件"路徑"
    image1 = io.imread(uploaded_file, as_gray=True) # 從路徑讀取文件, 並轉成灰色(灰階轉換)
    # io.imread是skimage的一個函數，用於讀取圖像文件。

    # 轉為(28,28)影像
    image_resized = resize(image1, (28,28), anti_aliasing=True) # 開啟抗鋸齒功能，這有助於減少縮放過程中可能產生的圖像失真，特別是在減小圖像尺寸時。
    x1 = image_resized.reshape(1,28,28) # +維度1為1, 表批次

    # 反轉顏色, 顏色0為白色, 與RGB不同,RGB是黑
    x1 = torch.FloatTensor(1-x1).to(device) # 這裡把顏色反轉(1-這裡的值)

    # 預測
    predictions = torch.softmax(model(x1), dim=1) # dimension = 2維
    print(f'actual/predictions: {i} {np.argmax(predictions.detach().cpu().numpy())}') # 預測比例最大的=預測結果
    # .cpu()搬到cpu的原因: 需要將數據轉換成NumPy數組進行處理時，就需要將數據從gpu轉移到cpu
    # numpy不支持在gpu上的數據進行操作。因此當使用numpy處理tensor時，必須先將這些tensor轉移到cpu
'''
補充1: resize 與 reshape
 (1) reshape: 關注於改變數據的形狀而不改變數據的總量，不會改變數據的內容，只是改變數據的形狀(維度)
 (2) resize: 會改變數據的總量，添加新的數據點或移除現有的數據點。特別在圖像處理中，通過改變圖像的解析度來增加或減少像素的總數，從而達到縮放圖像的目的。

補充2: RGB 與 圖像處理的顏色
  (1) RGB: 用三個分量（紅色、綠色、藍色）的組合來表示顏色，每個分量的值通常在0到255之間
  (2) 圖像處理的顏色: 通常使用0到1之間的值來表示顏色的深淺，其中0代表黑色，1代表白色
 ''' 

# 顯示模型的彙總資訊 (1)
print(model)
''' 
執行結果: 包含每一神經層的名稱及輸出入參數個數
Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)  # 保留1維, 2維開始全部攤平
    (1): Linear(in_feature=784, out_feature=256, bias=True) # 輸入784, 輸出256, 包含bias(b)
    (2): Dropout(p=0.2, inplace=False) # 20% 神經元被隨機dropout, inplace=False: 創建並返回一個新的張量來存儲輸出結果，而原始輸入數據不會被修改
    (3): Linear(in_feature=256, out_feature=10, bias=True) # 輸入256, 輸出10, 包含b(interception)
) 

'''
# 顯示模型的彙總資訊 (2)
for name, model in model.named_children():
# model.named_children()：返回model中所有直接子模塊的名稱和實例(sequential、dropout、linear)
 # (1) 名稱: 沒有指定時, 會自動使用index (start from 0)
 # (2) 實例: 對應名稱的模塊實際的對象, 包含了模塊的所有數據和方法，包括層的權重、偏置、激活函數等。
    print(f'{name}:{model}')

# 顯示模型的彙總資訊 (3): 可以安裝 torchinfo 或 torch-summary 套件, 以更美觀
# pip install torchinfo
from torchinfo import summary
summary(model, (6000, 28, 28)) # input dimension size
''' 
summary(model, input_size) 會給出的結果:
 
 (1) 每層的類型（例如Linear、Conv2d）
 (2) 每層的輸出形狀
 (3) 每層的參數數量（包括權重和偏置）
 (4) 每層使用的激活函數
 (5) 模型總的參數數量, 分為可訓練的參數和不可訓練的參數。

'''
