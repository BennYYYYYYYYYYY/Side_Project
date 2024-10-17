'''
(問題6)
目前 MNIST 影像為單色灰階, 若是彩色可以辨識嗎? 如何修改?
解答: 可以, 若顏色有助於辨識, 可以接RGB三通道分別輸入辨識, 後面講解CNN時會有範例 
'''


'''
(問題7)
執行週期(epochs)設為5, 設成其他值會更好嗎?
將epochs改成10
結果: 準確率升成92%, 但看損失率的圖可以發現, 其實後面都降不下去了
理論上更多的訓練會讓準確率更高, 然而, 過多的訓練會造成overfitting而使準確率下降
'''
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = "" 
BATCH_SIZE = 1024  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

train_ds = MNIST(PATH_DATASETS, train=True, download=True, 
                 transform=transforms.ToTensor())

test_ds = MNIST(PATH_DATASETS, train=False, download=True, 
                 transform=transforms.ToTensor())

print(train_ds.data.shape, test_ds.data.shape)


model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 256), 
    nn.Dropout(0.2),   
    torch.nn.Linear(256, 10), 
).to(device)

epochs = 10   # 把執行週期改成10
lr=0.1

train_loader = DataLoader(train_ds, batch_size=600)
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

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        test_loss += criterion(output, target).item()
        
        pred = output.argmax(dim=1, keepdim=True)  
        
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
batch = batch_idx * len(data)
data_count = len(test_loader.dataset)
percentage = 100. * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count}' + 
      f' ({percentage:.0f}%)\n')

predictions = []
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i][0], test_ds[i][1]
        data = data.reshape(1, *data.shape).to(device)
        output = torch.argmax(model(data), axis=-1)
        predictions.append(str(output.item()))

print('actual    :', test_ds.targets[0:20].numpy())
print('prediction: ', ' '.join(predictions[0:20]))

from skimage import io
from skimage.transform import resize
import numpy as np

for i in range(10):
    uploaded_file = f'C:\\Users\\user\\Desktop\\Python\\pytorch data\\myDigits\\{i}.png'
    image1 = io.imread(uploaded_file, as_gray=True)

    image_resized = resize(image1, (28, 28), anti_aliasing=True)    
    X1 = image_resized.reshape(1,28, 28) #/ 255.0

    X1 = torch.FloatTensor(1-X1).to(device)

    predictions = torch.softmax(model(X1), dim=1)
    print(f'actual/prediction: {i} {np.argmax(predictions.detach().cpu().numpy())}')