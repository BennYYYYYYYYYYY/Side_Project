
# 採用部分模型(只萃取特徵, 不作辨識)

import torch
from torchvision import models
from torch import nn
from torchsummary import summary
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# 載入VGG 16 模型
model = models.vgg16(pretrained=True) 
model._modules # 查看 VGG-16 模型的所有層

class new_model(nn.Module): # 定義新的類別 new_model, 繼承自基本模型類別 nn.Module
    def __init__(self, pretrained, output_layer): # 初始化函數, 接受Pretrained 和中斷層名稱 (output_layer) 為參數
        super().__init__() # 調用父類別(nn.Module)的初始化函數
        self.output_layer = output_layer
        self.pretrained = pretrained
        self.children_list = [] 
        # 依序取得每一層
        for n,c in self.pretrained.named_children(): # 迭代預訓練模型的每一子模組, named_children() 返回生成器, 包含子模組的名字和實體
            self.children_list.append(c) # 將當前子模組添加到列表中
            # 找到特定層即終止
            if n == self.output_layer: # 檢查當前子模組的名字是否與指定的中斷層名稱相匹配
                print('found !!') 
                break

        # 建構新模型
        self.net = nn.Sequential(*self.children_list) # Sequential 建立一個新的連續模組, 包含從頭到指定層的所有子模組
        self.pretrained = None # 釋放原先預訓練模型的引用
        
    def forward(self,x): # 前向傳播函數
        x = self.net(x) # 將輸入 x 通過定義好的Sequential
        return x
    
model = new_model(model, 'avgpool') # 創建 new_model 的實例
# 使用原始的 VGG-16 模型和指定的中斷層 'avgpool'

model = model.to(device)   

model._modules # 查看新定義模型的模組

# 任選一張圖片，例如老虎側面照，取得圖檔的特徵向量
from PIL import Image
from torchvision import transforms

# 使用 PIL 的 Image.open 方法打開圖片文件
filename = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_test\\tiger2.jpg'
input_image = Image.open(filename) 

transform = transforms.Compose([ # 變換序列
    transforms.Resize((224, 224)), # 224x224 像素
    transforms.ToTensor(), # 轉為 tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 標準化圖片的每個顏色通道 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image) # 將定義的圖片變換應用於打開的圖片
input_batch = input_tensor.unsqueeze(0).to(device) # 增加一維(筆數)

# 預測
model.eval() 
with torch.no_grad():
    output = model(input_batch)
output    

print(output.shape) # 打印輸出張量的形狀, 確認模型輸出的維度

import os
from os import listdir # listdir: 列出給定目錄下的所有檔案和目錄名
from os.path import isfile, join # isfile: 檢查給定路徑是否是一個文件
# join: 將多個路徑組件合併成一個完整的路徑

'''
使用 os.listdir 函數時, 這個函數會列出指定目錄下的所有檔案和子目錄的名稱, 但這些返回的名稱不包括其前置的路徑。
也就是說, 它們僅僅是檔案或子目錄的名稱, 而不是從根目錄或當前目錄到該檔案或目錄的完整路徑。
如果想對這些檔案進行任何檔案系統操作(如打開、讀取、檢查檔案是否存在等), 需要使用完整的路徑來指定這些檔案

os.path.join 函數將 img_path 和 'xxxx.jpg' 正確地合併成一個完整的路徑, 使能夠正確地打開位於指定目錄下的檔案
'''
# 取得 images_test 目錄下所有 .jpg 檔案名稱
img_path = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_test'
image_files = np.array([f for f in listdir(img_path)  # 檢索指定目錄下所有以 '.jpg' 結尾的文件, 將其儲存為 NumPy 陣列
        if isfile(join(img_path, f)) and f[-3:] == 'jpg'])
# listdir(img_path) 列出 img_path 指定的目錄下的所有項目
# isfile(join(img_path, f)) 檢查由 join(img_path, f) 返回的完整路徑是否是一個文件, 使用 join 是為了確保路徑的正確性
# f[-3:] == 'jpg' 確保檔案名稱的最後三個字符是 'jpg', 意味著它是 JPEG 格式的圖片檔案
image_files


# 合併所有圖檔
model.eval()
X = torch.tensor([]) 
for filename in image_files:
    input_image = Image.open(os.path.join(img_path, filename)) # 確保路徑正確性, 使得腳本能夠從正確的目錄位置讀取每一個圖片檔案
    input_tensor = transform(input_image) # 套用預定義的轉換
    input_batch = input_tensor.unsqueeze(0).to(device) # 增加一維(筆數)
    if len(X.shape) == 1: # 檢查 X 的維度, 空的張量 torch.tensor([]) 被視為一維[shape為(0)]
        # 一維數據, 經過len把它輸出長度: 例如(0)會變成1, 所以意味著這個張量目前沒有存儲任何數據
        X = input_batch 
    else: # 如果 X 不是一維的 (已包含數據)
        X = torch.cat((X, input_batch), dim=0) # torch.cat 函數用於將一系列張量沿指定的維度拼接起來
        # X, input_batch: 這些是要合併的兩個張量
        # dim=0: 這指定了拼接的維度

# 預測所有圖檔
with torch.no_grad():
    features = model(X) # model(X) 進行了所有圖片的特徵提取, 這些特徵將用於後續的相似度計算
features.shape # 查看獲得的特徵張量的形狀

from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarity: 用於計算兩組數據之間的餘弦相似度

# 比較 Tiger2.jpg 與其他圖檔特徵向量
no=-2 # 定義變數 no 為 -2, 用於指定要比較的圖片索引 (從列表末尾倒數第二個)
print(image_files[no]) # 打印出要比較的圖片文件名, 根據 no 的索引

# 轉為二維向量，類似扁平層(Flatten)
features2 = features.cpu().reshape((features.shape[0], -1)) # 二維數據用-1自動計算

# 排除 Tiger2.jpg 的其他圖檔特徵向量
other_features = np.concatenate((features2[:no], features2[no+1:])) # np.concatenate(): 用於合併多個陣列
# features2[:no] 和 features2[no+1:] 表示從特徵陣列 features2 中取出除了索引 no 以外的所有特徵向量
# features2[:no] 表示從陣列 features2 的開頭取到索引 no 之前的所有元素
# features2[no+1:] 表示從陣列 features2 中索引 no+1 開始取到陣列的結尾

# 使用 cosine_similarity 計算 Cosine 函數
similar_list = cosine_similarity(features2[no:no+1], other_features, 
                                 dense_output=False)
# cosine_similarity: 用於計算兩組數據點之間的 Cosine 相似度, 用於評估它們的相似性
# features2[no:no+1] 取出 features2 中索引 no 的特徵向量, 這是計算相似度的基準向量
# other_features 是與基準向量計算相似度的向量集
# dense_output=False: cosine_similarity 函數會返回一個稀疏矩陣 (sparse matrix) 而不是一個密集矩陣 (dense matrix) 這在數據矩陣很大時可以節省記憶體

# 顯示相似度，由大排到小
print(np.sort(similar_list[0])[::-1]) # np.sort() 函數對 similar_list 中的第一行（即所有計算的相似度）進行排序
# [::-1]: 用於反轉數組，使得數據從大到小排列

# 依相似度，由大排到小，顯示檔名
image_files2 = np.delete(image_files, no) # np.delete(image_files, no) 從 image_files 陣列中移除索引為 no 的檔名
# 計算與這個基準向量的相似度時，需要從特徵集中排除這個基準向量自身

image_files2[np.argsort(similar_list[0])[::-1]] # np.argsort() 函數返回的是數組值從小到大的索引排序, 並再次使用[::-1]使變成大到小


# 比較對象：bird.jpg
no=1
print(image_files[no]) # 打印 image_files 陣列中索引為 1 的元素


# 使用 cosine_similarity 計算 Cosine 函數
other_features = np.concatenate((features2[:no], features2[no+1:])) # 第一個~no前一個 concatenate no+1~最後一個
# other_features 通過合併 features2 陣列中，除了索引 no 的所有特徵向量來創建
similar_list = cosine_similarity(features2[no:no+1], other_features, 
                                 dense_output=False) 

# 顯示相似度，由大排到小
print(np.sort(similar_list[0])[::-1])

# 依相似度，由大排到小，顯示檔名
image_files2 = np.delete(image_files, no)
image_files2[np.argsort(similar_list[0])[::-1]]

'''
cosine_similarity(X, Y=None, dense_output=True)
X, Y: 這兩個參數都是特徵陣列。
X 是必須提供的, 而 Y 是可選的。

如果只提供 X, 函數將計算 X 中所有向量的兩兩相似度。
如果同時提供 X 和 Y, 則函數會計算 X 中每個向量與 Y 中每個向量之間的相似度。


X 和 Y 應該是形狀為 (n_samples_X, n_features) 和 (n_samples_Y, n_features) 的二維數組。
其中 n_samples 表示樣本數量, n_features 表示特徵數量
'''
