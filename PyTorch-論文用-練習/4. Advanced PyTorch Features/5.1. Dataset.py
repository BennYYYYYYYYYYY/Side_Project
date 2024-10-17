import os 
import torch
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下載MNIST手寫阿拉伯數字資料
train_ds = MNIST('', train=True, download=True,
                 transform=transforms.ToTensor())
# transforms.ToTensor(): 將PIL圖像或ndarray轉換為大小為(C x H x W)的torch.Tensor格式

test_ds = MNIST('', train=False, download=True,
                transform=transforms.ToTensor())
print(train_ds.data.shape, test_ds.data.shape)

# 顯示第一張圖片
import matplotlib.pyplot as plt

x = train_ds.data[0] # 第一筆
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()

training_data = FashionMNIST(
    root='data', # root參數指定了數據集的下載和存儲位置的根目錄。
    train=True,
    download=True,
    transform=transforms.ToTensor()
) # root='data': FashionMNIST數據集將被下載到當前工作目錄下的data子目錄中。如果該子目錄不存在，則會自動創建。

test_data = FashionMNIST(
    root='data',
    train=False,
    download = True,
    transform = transforms.ToTensor()
)

labels_map = {    # 定義數字標籤對應的類別名稱
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8)) # 創建一個新的圖形，指定圖形的大小為8x8英寸。
cols, rows = 3, 3
for i in range(1, cols*rows+1): # 循環將遍歷9個格子
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # torch.randint(): 生成一個隨機整數，範圍在0到len(training_data)-1之間
    # size=(1,): 指定生成隨機數的形狀，此例中生成一個數字。
    img, label = training_data[sample_idx] 
    figure.add_subplot(rows, cols, i) # 在當前圖形中加入一個子圖(行數, 列數, 子圖索引)
    plt.title(labels_map[label]) #  為子圖設置標題，在此是從 label_map 字典中根據 label 查找到的對應類別名稱。
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
    # .squeeze(): 去除張量形狀中所有維度為1的維度。以符合 imshow 的要求
plt.show()



from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import skimage

orig_img = skimage.data.astronaut() # 下載圖片
skimage.io.imsave('astronaut.jpg', orig_img)
# skimage.io.imsave: 將上一步獲取的圖像保存到文件系統中。 'astronaut.jpg' 文件名稱
plt.axis('off')
plt.imshow(orig_img) 

# 轉換輸入須為Pillow庫格式, 故以Pillow函數讀取
orig_img = Image.open('C:\\Users\\user\\Desktop\\Python\\PyTorch\\CH5\\astronaut.jpg')


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
#with_orig=True: 表示在繪製變換後的圖像時，也會在每行的開始處顯示原始圖像。
# **: 被用來捕獲函數調用中的所有未明確聲明的關鍵字參數，並將它們存儲在一個字典中。
# 這種機制允許在函數定義中使用靈活的參數，而不必提前知道所有可能被使用的參數名稱。
    if not isinstance(imgs[0], list):
    # isinstance: 檢查是否為xxx數據類型
        imgs = [imgs] # 如果imgs[0]不是list, 用另一個list把imag list包起來(2維list)

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig # with_orig: 加上原始圖片(+1)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False) # 創造整個圖形(fig)、子圖軸(axs)
    for row_idx, row in enumerate(imgs): # 每個row代表一組圖片
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row): # 再把每組row圖片的拿來跑
            ax = axs[row_idx, col_idx] # 選擇子圖
            ax.imshow(np.asarray(img), **imshow_kwargs)
            # np.asarray(img): 將img轉換成NumPy數組，以便imshow可以處理。
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[]) # 刻度空白
            # ax.set(): 可同時設定多個軸（Axes）屬性。

    if with_orig: # 檢查變量with_orig是否為True
        axs[0, 0].set(title='Original image') # 設置[0,0]子圖標題
        axs[0, 0].title.set_size(8) # 設置[0,0]子圖標題字體大小
    if row_title is not None: # 檢查變量row_title是否不是空的
        for row_idx in range(num_rows): 
            axs[row_idx, 0].set(ylabel=row_title[row_idx])
            # 把每一行第一列設定ylabel: 標籤文字從row_title列表中取得

    plt.tight_layout()
# plt.tight_layout(): 自動調整子圖參數，使之填充整個圖表區域，同時保持子圖之間的間隔適當，以避免標題、軸標籤等元素的相互重疊。

# resize
resized_imgs = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
# Resize: 用於調整圖像的尺寸, size=size指定了新圖像的尺寸, orig_img是被調整尺寸的原始圖像
# 原始圖像將被調整到30x30, 50x50, 100x100的尺寸，以及保留其原始尺寸
plot(resized_imgs)


center_crops = [T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
# T.CenterCrop: 中心裁剪
plot(center_crops)

# FiveCrop: 以左上、右上、左下、右下及中心點為參考點, 一次剪裁五張, 每個區域都將裁剪成100x100像素的大小
(top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(100, 100))(orig_img) # 一樣要保留原圖, 放最前面
plot([top_left, top_right, bottom_left, bottom_right, center])

# 轉灰階
gray_img = T.Grayscale()(orig_img) # T.Grayscale(): 將圖像轉換成灰階的轉換操作
plot([gray_img], cmap='gray')

# 旁邊補零: 指定補零寬度為3、10、30、50
padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
# T.Pad: 對原始圖像orig_img進行邊緣填充, padding參數指定了填充的量, 這裡padding的值在循環中變化，分別為3,10,30,50像素
plot(padded_imgs)

plt.show()

# 自訂資料集(Custom Dataset)
labels_code = {v.lower():k for k, v in labels_map.items()} # 從一個現有的字典 labels_map 創建一個新的字典 labels_code
# for k, v in labels_map.items(): 遍歷labels_map字典的每一個項目，其中k是原字典的鍵，v是原字典的值
# v.lower(): 將標籤v轉換為全部小寫
# {v.lower(): k for ...}: 創建一個新的字典，其中以小寫的標籤名作為鍵，原來的鍵作為值。

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import re # 正則表達式（Regular Expression）模塊
from torchvision import transforms, datasets

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = [file_name for file_name in os.listdir(img_dir)] # 讀取img_dir目錄下的所有文件名，並將它們存儲在img_labels列表中
        self.img_dir = img_dir # 圖像目錄的路徑存儲在實例中
        self.transform = transform # 轉換儲存在實例中
        self.target_transform = target_transform # 轉換儲存在實例中

    def __len__(self): # 返回數據集中的項目總數
        return len(self.img_labels)
    
    def __getitem__(self, idx): # 接受一個索引(idx)，並返回該索引對應的圖像及其標籤
        img_path = os.path.join(self.img_dir, self.img_labels[idx]) # os.path.join: 用於將多個路徑組件合併成一個完整的路徑字符串。
        # 從 self.img_labels 列表中選取索引為 idx 的元素
        image = read_image(img_path) # 加載圖像文件
        label = self.img_labels[idx].split('.')[0] # 將文件名以.為分隔符進行分割，並取第一部分作為標籤
        label = re.sub('[0-9]', '', label) # 將所有數字替換為空字符串,只留下字母部分, label:要進行搜索和替換的原始字符串

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        image = image.reshape(*image.shape[1:]) # 去除一維資料(3維轉2維)
        image = 1.0-image # # 反轉顏色，顏色0為白色，與RGB色碼不同，它的0為黑色
        label = labels_code[label.lower()] # 從先前定義的字典labels_code中根據轉換成小寫的label查找對應的值

        return image, label # 返回圖像和標籤
    
# 模型載入
model_path = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\CH4\\FashionMNIST.pt'
model = torch.load(model_path)

# 建立transforms
transform = transforms.Compose([
    transforms.Grayscale(), # 將圖像轉換成灰階
    transforms.Resize((28,28)), # 將圖像大小調整為 28x28 像素
    transforms.CenterCrop(28), # 從圖像中心裁剪出 28x28 的區域
    transforms.ConvertImageDtype(torch.float), # 將圖像的數據類型轉換為浮點
])

# 建立DataLoader
train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, shuffle=False, batch_size=10)


model.eval()
criterion = nn.CrossEntropyLoss()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.view_as(pred)).sum().item() # 計算預測正確的數量

# 計算平均損失和正確率
test_loss /= len(test_loader.dataset)
data_count = len(test_loader.dataset)
percentage = 100. * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count}' + 
      f' ({percentage:.0f}%)\n')

data, target = next(iter(test_loader))
print(data.shape, target)
# iter(test_loader) 創建了一個迭代器
# 可以使用 next() 函數來逐個獲取 test_loader中的元素。每個元素是一個批次的數據。

