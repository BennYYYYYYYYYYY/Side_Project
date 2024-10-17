'''
可解釋的人工智慧（Explainable AI，簡稱XAI）是一種旨在讓人工智慧（AI）的決策過程更為透明和可理解的技術。
在傳統的AI應用中，尤其是那些基於深度學習和大數據的模型，其決策過程往往被視為“黑盒”，外部觀察者難以理解其內部運作和決策依據。
XAI的目標是改變這一狀況，通過提供解釋來幫助用戶理解模型的行為。

實現XAI的方法有很多，包括但不限於：

模型透明度：選擇本質上具有高透明度的模型，如決策樹，其決策過程容易追蹤和解釋。

後設解釋：為黑盒模型（如深度神經網絡）提供附加的解釋層或工具，比如局部可解釋模型-解釋技術（LIME）或集成梯度。

視覺化工具：利用各種數據視覺化技術來解釋和展示模型的行為，例如通過特徵重要性圖來顯示哪些輸入特徵對模型預測最有影響力。
'''
# pip install torchsummary

import torch
from torchvision import models
from torch import nn
import numpy as np
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

rn18 = models.resnet18(pretrained=True) # 載入一個預訓練的 ResNet-18 模型
# pretrained=True 表示模型已經在 ImageNet 數據集上訓練過
# 使得模型能夠在沒有從頭開始訓練的情況下直接用於圖像識別或進行微調(fine-tuning)

children_counter = 0
for n,c in rn18.named_children():
# named_children() 是一個生成器，遍歷模型的直接子模塊，並返回每個子模塊的名稱 (n) 和該子模塊本身 (c)
    print("Children Counter: ",children_counter," Layer Name: ",n)
    # 在每次迭代中，打印出當前子模塊的索引, 子模塊的名稱
    children_counter+=1 # 計數器+1

rn18._modules
# 這行代碼獲取模型的所有模塊的有序字典。這個字典的鍵是子模塊的名稱，值是子模塊本身。

'''
有序字典(Ordered Dictionary): 這個資料結構。
它是一種特殊的字典，與一般的字典相比，有序字典會根據元素被添加進字典的順序來保持元素的次序。
'''

from torchsummary import summary # 用於顯示模型的摘要信息

# 模型 rn18 的摘要
summary(rn18.to(device), input_size=(3, 224, 224)) # 模型首先被移動到指定的設備，並設定輸入大小為 (3, 224, 224)
# 這是標準的 ImageNet 圖像尺寸( 3 表示 RGB 三個顏色通道)

class new_model(nn.Module): # 定義新的類 new_model，繼承自 nn.Module
    def __init__(self, output_layer): # 
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = models.resnet18(pretrained=True) # 加載預訓練的 ResNet-18 模型
        self.children_list = []
        # 依序取得每一層
        for n,c in self.pretrained.named_children(): # named_children() 遍歷預訓練模型的所有子模塊
            self.children_list.append(c) # 將每個子模塊添加到 children_list 中
            # 找到特定層即終止
            if n == self.output_layer: # 直到找到名為 output_layer 的層為止
                print('found !!')
                break 

        # 建構新模型
        self.net = nn.Sequential(*self.children_list) # 使用 children_list 中的模塊建構一個新的連續模型
        # 使用星號 (*) 展開列表，意味著將列表中的每個元素都作為獨立的參數傳遞給 nn.Sequential

        self.pretrained = None # 釋放原始的預訓練模型，以節省記憶體
        # 將 self.pretrained 設為 None 可以幫助確保原始的完整預訓練模型不再佔用寶貴的記憶體資源，尤其是在已經將需要的子模塊取出後
        
    def forward(self,x): # 向前傳播
        x = self.net(x)
        return x
    
model = new_model(output_layer = 'layer1') # 新模型將包含從 ResNet-18 的輸入層開始直到 layer1 為止的所有層
# 在 new_model 的構造函數中，它會遍歷原始 ResNet-18 模型的層，一旦達到 layer1 層，迭代就會停止，並將這部分的層添加到新模型中
model = model.to(device)    

summary(model,input_size=(3, 224, 224)) # 顯示模型的結構和參數
# 模型預期的輸入圖像大小為 224x224 像素，並且有三個顏色通道 (紅、綠、藍)

from PIL import Image # Python Imaging Library，用於圖片的開啟、處理等操作
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms # 一系列的圖像變換操作

img = Image.open("C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_test\\cat.jpg") # 使用 Image.open 加載圖片
plt.imshow(img) # 顯示圖片
plt.axis('off') # 去掉坐標軸
plt.show()

resize = transforms.Resize([224, 224])
img = resize(img) # 將圖片大小調整為 224x224 像素, (這是模型預期的輸入尺寸)

to_tensor = transforms.ToTensor() # 格式轉換成 FloatTensor, 並且將像素值範圍從 [0, 255] 正規化到 [0.0, 1.0]
img = to_tensor(img).to(device)
img = img.reshape(1, *img.shape) # 增加一個批次維度, 因為模型預期的輸入是一批圖片
out = model(img) # 丟進model 
out.shape # 顯示輸出的形狀，對於理解模型的輸出層結構很有幫助

def show_grid(out): # 把model output丟進去
    square = 8 # 定義每行和每列顯示的特徵圖數量
    plt.figure(figsize=(12, 10)) # 創建一個圖形，尺寸為 12x10 英寸
    for fmap in out.cpu().detach().numpy(): # out.cpu() 將模型輸出從 GPU 轉移到 CPU
    # 這個循環對每一個特徵圖（fmap）進行迭代處理

        # plot all 64 maps in an 8x8 squares
        ix = 1
        # 共創建 square * square  (64)個子圖
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix) # subplot 函數用於在當前圖形中創建一個子圖，位置由 ix 指定
                ax.set_xticks([]) # 移除 x 軸的刻度
                ax.set_yticks([]) # 移除 y 軸的刻度
                # plot filter channel in grayscale
                plt.imshow(fmap[ix-1, :, :], cmap='gray') # 以灰階顯示該子圖
                # fmap[ix-1, :, :] 指定要顯示的特徵圖部分
                # fmap[ix-1, :, :] 表示從 fmap 陣列中選取第 ix-1 個特徵圖的所有行和列(:)表所有
                ix += 1 # 更新子圖索引
        # show the figure
        plt.show()
        
show_grid(out) # 使用show_grid()

model = new_model(output_layer = 'layer2').to(device) # 創建一個新模型，指定輸出層為 'layer2'，並將模型移至適當的設備
out = model(img) # 使用模型對圖片進行推理，獲得輸出
show_grid(out) # 用 show_grid 函數展示輸出的特徵圖

model = new_model(output_layer = 'layer3').to(device) # 更改輸出層為 'layer3'
out = model(img) 
show_grid(out)  

model = new_model(output_layer = 'layer4').to(device) # 更改輸出層為 'layer4'
out = model(img)
show_grid(out)

'''
這個程序是通過逐層改變模型的輸出，來觀察不同層的特徵激活情況。
這在理解和分析深度學習模型的內部工作原理方面非常有用。
每次改變輸出層後，都可以看到該層的不同特徵圖如何反映圖像的不同特性。
這有助於了解每一層在圖像處理中的作用和貢獻。
'''
