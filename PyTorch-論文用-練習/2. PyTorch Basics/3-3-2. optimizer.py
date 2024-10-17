'''
1. 定義訓練函數: reset(optimizer.zero_grad()) -> backward -> update(optimizer.step()) -> append
 (1) 定義優化器: 之前為固定的lr, 改用Adam的優化器他會採取動態衰減(decay)的lr
 (2) 梯度重置: 改由優化器控制, optimizer.zero_grad() 取代 model.zero_grad()
 (3) 權重更新: 改用optimizer.step()取代w,b的逐一更新
'''
import torch
import numpy as np
n = 100
X = np.linspace(0, 50, n)
y = np.linspace(0, 50, n)
X += np.random.uniform(-10, 10, n) 
y += np.random.uniform(-10, 10, n)
def create_model(input_feature, output_feature): # 輸入特徵數量, 輸出特徵數量
    model = torch.nn.Sequential( # 創建一個順序模型
        torch.nn.Linear(input_feature, output_feature), # 順序1: Linear神經層(輸入特徵數量, 輸出特徵數量)
        torch.nn.Flatten(0, -1) # 扁平層: 把全部壓成一維(全部乘起來)
    )
    return model

def train(X, y, epochs=100, lr=1e-4):
    model = create_model(1, 1) # 模型期望收到1個特徵並輸出1個特徵 
    loss_fn = torch.nn.MSELoss(reduction='sum')
    # 定義優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # torch.optim.Adam(): 創造Adam優化器
    # 參數: 優化器需要更新的參數，即所有可訓練的權重與偏差: model.parameters()為返回模型參數的跌代器 
    loss_list, w_list, b_list = [], [], []
    for epoch in range(epochs):
        y_pred = model(X) # x丟入模型, 會輸出1個特徵
        MSE = loss_fn(y_pred, y)  # y_pred, y 放入MSE損失函數
        optimizer.zero_grad() # 梯度重置: 改由優化器控制
        MSE.backward() # 計算
        optimizer.step() # 權重更新, 其中pytorch自己會搞定w,b的初始化
        # 紀錄訓練結果
        if (epochs+1) % 1000 == 0 or epochs < 1000:
            w_list.append(model[0].weight[:, 0].item()) #第一個模型[0]的權重w
            b_list.append(model[0].bias.item())
            loss_list.append(MSE.item())
        return w_list, b_list, loss_list
x2, y2 = torch.FloatTensor(X.reshape(X.shape[0], 1)), torch.FloatTensor(y)
w_list, b_list, loss_list = train(x2, y2)
print(f'w={w_list[-1]}, b={b_list[-1]}')
            

'''
2. 除了回歸外，也可以處理分類(Classification)問題
資料集採用Scikit-learn套件內建的Iris
'''
import pandas as pd
from sklearn import datasets
# dataset.data: 數據特徵
# dataset.target: 需要預測出的目標

dataset = datasets.load_iris() # 從scikit-learn下載iris數據集
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
# 創建DataFrame, 放入iris數據特徵(150, 4)數組, 並指定columns的title(dataset.feature_names)
print(df.head())
# 顯示前5 row的數據(.head)

# 把資料集分成訓練、測試集，測試資料佔20%
from sklearn.model_selection import train_test_split # 函數用於分割資料
X_train, X_test, y_train, y_test = train_test_split(df.values, # 把df(DataFrame)轉成Numpy數列
                                                   dataset.target, test_size=0.2) # 參數指定測試資料佔20%
                                            # 目標值（Target）: 指的是想要預測或分類的數據(答案), 所以dataset.target就是答案清單

# 進行 one-hot encoding: y變成3個變數，代表三個品種的機率
''' 
one-hot encoding: 將類別型數據轉換為一組由0和1組成的向量，
每個類別由一個獨立的向量表示，該向量中的一個元素為1，其餘為0。
'''
y_train_encoding = pd.get_dummies(y_train) # 把答案(y) encode 成 one-hot編碼
y_test_encoding = pd.get_dummies(y_test) # dummy variable(啞變量/虛擬變量)

# one-hot encoding 也可以用PyTorch做
torch.nn.functional.one_hot(torch.LongTensor(y_train)) # 轉變為LongTensor(one-hot編碼要輸入整數)
# torch.nn.functional.one_hot: 函數將類別資料轉換為one-hot編碼形式

# 轉成Tensor
X_train = torch.FloatTensor(X_train)
y_train_encoding = torch.FloatTensor(y_train_encoding.values)
X_test = torch.FloatTensor(X_test)
y_test_encoding = torch.FloatTensor(y_test_encoding.values)

# 建立神經網路模型
model = torch.nn.Sequential(
    torch.nn.Linear(4, 3), # 模型期望輸入4個特徵，且輸出3個特徵(預測值)
    torch.nn.Softmax(dim=1) # dim: 對哪維度使用，dim=1即對2維資料使用(row)
)
# Softmax: 它將模型輸出的原始分數（也稱為logits）轉換為概率分布。
# 使得每個類別的概率都在0到1之間，且所有類別的概率之和為1

# 定義損失函數、優化器: 
loss_fn = torch.nn.MSELoss(reduction='sum') # reduction: 將多個損失值"減少"至1個的方法
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #參數: 優化器需要更新的參數

# 訓練模型
epochs = 1000
accuracy = []
losses = []
for i in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train_encoding)
    accuracy.append((np.argmax(y_pred.detach().numpy(), axis=1) == y_train).sum()/y_train.shape[0]*100) #(argument of the maximum)
# y_pred.detach(): 把y_pred從計算圖中分離(不會被追蹤)
# y_pred.detach().numpy(): 再把分離出來的Tensor轉成Numpy數列
# np.argmax( , axis=1): 用於找出指定軸上的最大值的"索引", 此為找二維(row,特徵)最大值的index
# 上面一大坨 == train: 預測索引的數組，與真實標籤的數組y_train進行比較(==) 相同返回True 不同False, 最終得到booling數組
# booling數組.sum(): True=1, False=0
# y_train.shape[0]: 獲得y_train的一維大小(即資料數量)
# 最終: (預測正確有幾個/訓練資料總數)*100 = "準確性"(accuracy)    
    losses.append(loss.item()) # MSE(sum)

    # reset
    optimizer.zero_grad()
    # backward
    loss.backward()
    # step
    optimizer.step()
    # append
    if i%100 == 0: # 若i能被100整除(每100次印出一次)
        print(loss.item()) # 印出損失函數值

# 繪製訓練過程的損失(losses)、準確率趨勢圖(accuracy)
import matplotlib.pyplot as plt
# fix中文亂碼
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif']=['Microsoft JhengHei'] #無襯線字體為微軟正黑體
plt.rcParams['axes.unicode_minus']=False #負號不用unicode

plt.figure(figsize=(12,6)) # 設置圖形大小: figsize(長,高)
plt.subplot(1, 2, 1) # 建了一個1行2列的子圖布局，並指定接下來的繪圖命令將作用於第一個子圖上
# plt.subplot(): 在一個窗口(figure)中創建多個圖表(子圖表)
# (所有子圖的行數, 所有子圖的列數, 要指定的索引號)
# 索引號: 由左至右，由上到下, 由"1"開始!!!! (不是0!!!!)
plt.title('損失', fontsize=20) 
plt.plot(range(0, epochs), losses) # X軸:跌代次數, y軸:損失
plt.ylim(0, 100) # (y limit):設定圖表中y軸的顯示範圍，此為從0到100。
# 繪製百分比數據時特別有用，因為它確保y軸可以完整地展示0%~100%的範圍

plt.subplot(1, 2, 2) # 選第二個子圖表
plt.title('準確性', fontsize=20)
plt.plot(range(0, epochs), accuracy)
plt.ylim(0, 100)
plt.show()

# 模型評估
predict_test = model(X_test)
_, y_pred = torch.max(predict_test, 1)
# torch.max(資料, 尋找維度): 尋找predict_test中每個樣本的預測得分最大值，並返回其索引
# 他會返還兩個值: (1) 最大值本身 (2)對應的索引
# (1): 用_接收,代表我不在乎他, (2): 用y_pred接收
print(f'測試資料準確度:{((y_pred.numpy() == y_test).sum()/y_test.shape[0]):.2f}')
# y_pred.numpy() == y_test：將預測索引y_pred轉換為numpy數組，與真正的y_test進行比較(返回booling)
# .sum()/y_test.shape[0]：計算預測正確的樣本數，並除以總樣本數y_test.shape[0]，得到準確率
# :後指定格式化語法, .2f 表示: 轉為保留兩位(2)小數的浮點數(f)

    

