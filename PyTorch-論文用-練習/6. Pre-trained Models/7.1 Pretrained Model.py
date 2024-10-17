'''


應用Pre-tained Model有三種方式
 1. 採用完整模型, 如果目標與模型訓練的資料相同
 2. 採用部分模型, 只萃取特徵, 不作輸出
 3. 採用部分模型, 並接上自訂輸入層與完全連接層, 可作預訓練模型資料以外的輸出
'''

# 採用完整模型

import torch
from torchvision import models # models 子模塊提供了預先訓練的模型, 如 ResNet、VGG 等
from torch import nn
import numpy as np
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

model = models.vgg16(pretrained=True) # 加載一個預訓練的 VGG-16 模型, pretrain=True代表使用訓練好的參數

children_counter = 0
for n,c in model.named_children(): # 打印出每個層的名稱, 索引
    print("Children Counter: ",children_counter," Layer Name: ",n)
    children_counter+=1 # 計數器+1

model.modules # 獲取一個模型中的所有模塊的迭代器，這包括模型本身以及其所有子模塊和嵌套的子模塊

torch.nn.Sequential(*list(model.children())[:]) # 創建 Sequential Model，包含了原模型中的所有子模塊。
# model.children() 返回的迭代器轉換成一個列表, 並使用切片操作 [:] 來獲取這個列表的全部元素
# * 用於解包列表, 意味著它把列表中的每個元素作為獨立的參數傳遞給 Sequential
'''
使用 * 來解包層列表並傳遞給新的 torch.nn.Sequential 容器時
不需要關心每個層的具體類型或功能 (卷積層、全連接層...)。
Sequential 容器將自動按照提供層的順序將它們串接起來，確保數據可以依次通過這些層。
'''

model._modules.keys() # 返回一個包含模型所有子模塊名稱的字典鍵 (keys) 的迭代器
# _module: 以字典的形式存儲了模型中所有直接子模塊的名稱和對應的模塊對象

model.features # 這是 VGG 模型中的一個屬性, 包含了模型的所有卷積層

model.features[0] # 獲取 features 模塊的第一個層
# features 屬性通常包含了模型的前幾層, 這些層主要負責從輸入影像中提取特徵

model.classifier[-1].weight.shape # 獲取分類器部分最後一層 (通常是一個全連接層) 的權重矩陣的形狀
# .weight：這是該層的權重參數, 是一個包含可學習參數的張量
# 針對分類任務設計的模型, classifier 屬性包含了模型的最後幾層, 這些層通常用於根據特徵進行最終的類別預測

model.classifier[-1].out_features # 返回分類器中最後一層的輸出特徵數 (最終的類別數)
# 使用預訓練模型時, 對應於 ImageNet 的類別數 (1000類) 

model = model.to(device) 

summary(model, input_size=(3, 224, 224)) # 模型摘要, 指定模型預期的輸入維度: 3個通道(RGB), 224x224像素

from PIL import Image
from torchvision import transforms

filename = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_test\\cat.jpg'
input_image = Image.open(filename) # 打開位於指定路徑的圖像

transform = transforms.Compose([  # 圖形轉換步驟: 包含3步驟
    transforms.Resize((224, 224)),  # 將圖像大小調整為 224x224 像素
    transforms.ToTensor(), # 將 PIL 圖像轉換成 PyTorch 張量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 標準化圖像數據
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image) # 將上述定義的轉換應用於圖像
input_batch = input_tensor.unsqueeze(0).to(device) # 加一個批次維度，因為模型期望的輸入通常是一批圖像
# unsqueeze(0) 用於在指定位置增加一個新的維度, 且為1

model.eval()
with torch.no_grad(): # 不進行梯度下降 
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0) 
# 模型output取出第一項, 並用 softmax 函數轉換為概率值 (對每個類別的分數進行概率化處理), dim=0: 轉換第一維度
# output[0] 提取的是模型對第一個圖像的輸出, 是一個形狀為 [num_classes] 的 tensor (就是模型能夠識別的類別數)

print(probabilities) # 每個類別的機率

print(f'{torch.argmax(probabilities).item()}: {torch.max(probabilities).item()}') # 印出概率最高的類別索引和對應的概率值
# torch.max() 會從找出最大的元素

with open("C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\imagenet_classes.txt", "r") as f: # 打開 "imagenet_categories" 文件, 以讀取模式打開
    # 文件應包含所有 ImageNet 分類類別的列表

    categories = [s.strip().split(',')[0] for s in f.readlines()] # 生成一個列表, 包含文件中每一行的第一個 (類別名稱的列表)
    # f.readlines(): 讀取文件的全部內容, 並將每一行作為列表的一個元素返回
    # 先使用 strip() 方法移除字符串兩端的空格和換行符, 然後使用 split(',') 按逗號分割字符串
    # 取分割後的第一個元素 (類別名)

categories[torch.argmax(probabilities).item()]
# torch.argmax() 函數用於找出給定張量中最大元素的索引

filename = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_test\\tiger2.jpg'
input_image = Image.open(filename) # 打開圖檔

transform = transforms.Compose([  # 圖形順序transform
    transforms.Resize((224, 224)), # 改成224x224像素
    transforms.ToTensor(), # 轉成 Pytorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 一般化 
                         std=[0.229, 0.224, 0.225])
])

input_tensor = transform(input_image) # 套用 transform
input_batch = input_tensor.unsqueeze(0).to(device) # 加一個批次維度，因為模型期望的輸入通常是一批圖像

# 預測
model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# 模型output取出第一項, 並用 softmax 函數轉換為概率值 (對每個類別的分數進行概率化處理), dim=0: 轉換第一維度
# output[0] 提取的是模型對第一個圖像的輸出, 是一個形狀為 [num_classes] 的 tensor (就是模型能夠識別的類別數)

max_item = torch.argmax(probabilities).item() # 每個類別的機率
print(f'{max_item} {categories[max_item]}: {torch.max(probabilities).item()}') # 印出概率最高的類別索引和對應的概率值

with open("C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\imagenet_classes.txt", "r") as f: # 打開 "imagenet_categories" 文件, 以讀取模式打開
    # 取第一欄
    categories = [s.strip() for s in f.readlines()] # 以行讀取f資料後, 用strip()把空格刪掉後, 生成list

model = models.resnet50(pretrained=True).to(device) # 加載一個預訓練的 ResNet-50 模型

# 預測
filename = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_test\\cat.jpg'
input_image = Image.open(filename) 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device) 

model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
max_item = torch.argmax(probabilities).item()
print(f'{max_item} {categories[max_item]}: {torch.max(probabilities).item()}')
# categories[max_item] 從列表 categories 中取得對應索引的類別名稱

model = models.resnet50(pretrained=True).to(device)

# 預測
filename = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_test\\cat.jpg'
input_image = Image.open(filename)

transform = transforms.Compose([
    transforms.Resize(256), # 調整影像的大小，使最小邊為 256 像素, 同時轉換過程會自動調整影像的其他邊，以保持原始影像的寬高比不變。
    transforms.CenterCrop(224), # 從影像中心裁剪出 224x224 像素的區域
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device) 

model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
max_item = torch.argmax(probabilities).item() # 返回max index
print(f'{max_item} {categories[max_item]}: {torch.max(probabilities).item()}')

top5_prob, top5_catid = torch.topk(probabilities, 5) 
# torch.topk(): 用於返回給定 tensor 中最大的 k 個元素及其索引, 這裡 k=5 表示返回前五個
for i in range(top5_prob.size(0)): # 由於 top5_prob 是由 torch.topk() 函数產生, 請求的是前5個元素，所以大小是 5
# .size(0): 用於獲取 tensor 沿著指定維度 (在這裡中是維度 0) 的大小
    print(f'{categories[top5_catid[i]]:12s}:{top5_prob[i].item()}') 
    # :12s 確保類別名稱左對齊並占用至少 12 個字符的空間, 以便於閱讀
    

sum(probabilities.cpu().numpy())

probabilities.cpu().numpy().argsort()[-5:][::-1]
# argsort(): NumPy函數, 返回數組值從小到大的排序index
# [-5:]  argsort() 返回的index數組中提取最後五個元素的index(即最大的5個)
# [::-1] 對數組進行翻轉, 使最高概率的index改在最前面

np.array(categories)[probabilities.cpu().numpy().argsort()[-5:][::-1]]

filename = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_test\\tiger2.jpg'
input_image = Image.open(filename)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device) # 增加一維(筆數)

# 預測
model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
max_item = torch.argmax(probabilities).item()
print(f'{max_item} {categories[max_item]}: {torch.max(probabilities).item()}')

