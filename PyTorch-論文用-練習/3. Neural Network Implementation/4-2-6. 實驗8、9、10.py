'''
(問題8)
準確率accuracy可以達到100%?

解答: 很難達到100%, 除非是利用數學模型推導出來的模型
且由於訓練資料與驗證/測試資料不同 
'''


'''
(問題9)
如果要辨識其他物件, 程式要如何修改?

以 FashionMNIST 的資料集示範
'''
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import FashionMNIST # 載入本次的資料集

PATH_DATASETS = "" 
BATCH_SIZE = 1024  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

train_ds = FashionMNIST(PATH_DATASETS, train=True, download=True, # 改成FashionMNIST資料集
                 transform=transforms.ToTensor())
test_ds = FashionMNIST(PATH_DATASETS, train=False, download=True, 
                 transform=transforms.ToTensor())

print(train_ds.targets[:10]) # 前10筆target值

print(train_ds.data[0]) # 第一筆data值

data = train_ds.data[0].clone() # 複製
data[data>0]=1 # >0 的都改成1
data = data.numpy() # 轉numpy 數列

text_image=[]
for i in range(data.shape[0]):
    text_image.append(''.join(data[i].astype(str))) # 把str串一起
text_image

import matplotlib.pyplot as plt

x = train_ds.data[0] # 抓第一筆data

plt.imshow(x.reshape(28,28), cmap='gray') # cmap: 灰色

plt.axis('off') # 隱藏axis 

plt.show() 

model = torch.nn.Sequential(
    torch.nn.Flatten(), 
    torch.nn.Linear(28*28, 256), 
    nn.Dropout(0.2),
    torch.nn.Linear(256, 10), 
).to(device)

epochs = 5
lr=0.1

# 建立 DataLoader
train_loader = DataLoader(train_ds, batch_size=600)

# 設定優化器(optimizer)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

model.train()
loss_list = []    
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * batch_idx / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                  f'  Loss: {loss.item():.6f}')
            
import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')
plt.show()

test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE)

model.eval() # 評估
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        test_loss += criterion(output, target).item()
        
        pred = output.argmax(dim=1, keepdim=True)  # dim=1,最大值index
        
        correct += pred.eq(target.view_as(pred)).sum().item()

data_count = len(test_loader.dataset)
test_loss /= data_count # 數據的平均損失值

percentage = 100. * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count}' + 
      f' ({percentage:.0f}%)\n')

predictions = []
with torch.no_grad(): # 用訓練好的參數
    for i in range(20):
        data, target = test_ds[i][0], test_ds[i][1]
        data = data.reshape(1, *data.shape).to(device) # +1維
        output = torch.argmax(model(data), axis=-1)
        predictions.append(str(output.item()))

print('actual    :', test_ds.targets[0:20].numpy())
print('prediction: ', ' '.join(predictions[0:20]))

import numpy as np

i=17
data = test_ds[i][0]
data = data.reshape(1, *data.shape).to(device)
#print(data.shape)
predictions = torch.softmax(model(data), dim=1)
print(f'0~9預測機率: {np.around(predictions.cpu().detach().numpy(), 2)}')
print(f'0~9預測機率: {np.argmax(predictions.cpu().detach().numpy(), axis=-1)}')

x2 = test_ds[i][0] 
plt.imshow(x2.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show() 

torch.save(model, 'FashionMNIST.pt')

label_dict = {   # 將label(數字)映射到對應的類別
    0:'T-shirt', 
    1:'Trouser',
    2:'Pullover', 
    3:'Dress', 
    4:'Coat',
    5:'Sandal', 
    6:'Shirt', 
    7:'Sneaker', 
    8:'Bag', 
    9:'Ankle boot'
}

from skimage import io
from skimage.transform import resize
import numpy as np
import os
 

test_data_folder = 'C:\\Users\\user\\Desktop\\Python\\PyTorch\\CH4\\FashionMNIST' # 指定存放數據的文件夾()
for file_name in os.listdir(test_data_folder):
# os.listdir() 函數接受一個路徑作為參數，並返回該路徑下所有檔案和目錄名的列表。
    image1 = io.imread(os.path.join(test_data_folder, file_name), as_gray=True)
    # os.path.join(): 用於將多個路徑合併成一個完整的路徑
    image_resized = resize(image1, (28, 28), anti_aliasing=True)  # 圖片縮為(28, 28)

    x1 = image_resized.reshape(1, 28, 28) 

    x1 = torch.FloatTensor(1-x1).to(device) # 反轉顏色

    predictions = torch.softmax(model(x1), dim=1) # 把label做機率表示
    
    print(f'actual/prediction: {file_name.split(".")[0]}/{label_dict[np.argmax(predictions.detach().cpu().numpy())]}')



'''
(問題10)
如果要辨識多個數字, 例如輸入4位數字, 如何辨識?

解答: 
 (1) 可以使用影像處理分割數字, 再分別依序輸入模型預測
 (2) 將視覺介面(UI)設計成四格, 規定使用者只能在每個格子內輸入一個數字
'''