''' 
3-2 使用運用自動微分解決的都是簡單線性回規，然而神經網路是由多條回歸線組成的
並且每條回歸線可能再乘上非線性的"Activation Function"

PyTorch可以直接建構各式各樣的神經層函數，我們只需專注在演算法的設計

PyTorch提供十多類神經層，每一類又有更多神經層，定義在torch.nn的空間下
'''


'''
1. 完全神經層(Linear Layer)
上一層神經層的每一個神經元都連結到下一層的每一個神經元
 (1) Linaer(完全連接): 基本的線性變換功能
 (2) Bilinear(雙線性): 考慮兩個輸入之間的交互作用，使模型能夠捕捉更複雜的關係
 (3) LazyLinear(懶加載線性): 參數的初始化延遲到首次接收到輸入數據時，根據輸入數據的維度動態配置
'''
import numpy as np
import torch
input = torch.randn(128, 20) # 輸入資料 = (128,20)的隨機常態分佈數值張量
# 128個樣本，每個樣本有20個特徵的輸入數據
print(input.shape) # 查看張量形狀 (128, 20)

# 建立神經層
# Linear參數: 
# (1) 輸入特徵個數
# (2) 輸出特徵個數 
# (3) 是否有偏差項(Bias)  
# (4) 裝置(None,'cpu','cuda')
# (5) 資料型態
# (6) Linear神經層的轉換為y=x(A.T)+b 
#   (6-1) x是輸入數據，A是層的權重，b是偏差項，y是輸出數據

layer1 = torch.nn.Linear(20, 30)  # 設定輸入特徵數為20，輸出特徵數為30
# 神經層計算: 未訓練的Linear就是執行內積: 維度(128, 20)@(20, 30)
output = layer1(input)
# 經過轉換後，output張量的形狀為(128, 30): 128筆資料, 30個特徵
print(output.shape) 


'''
2. 測試Bilinear神經層: 對兩組輸入進行雙線性變換。

不同於Linear只接收一組輸入，雙線性層接收兩組輸入。
這種層特別適合於需要考慮兩個不同輸入之間交互作用的場景，

Bilinear有兩個輸入神經元個數, 轉換為:y=(x1.T)(A.T)x2+b
'''
# 建立神經層
layer2 = torch.nn.Bilinear(20, 30, 40)
# 第一組輸入資料特徵量: 20
# 第二組輸入資料特徵量: 30
# 輸出特徵量: 40
input1 = torch.randn(128, 20) # 128筆資料, 20個特徵
input2 = torch.randn(128, 30) # 128筆資料, 30個特徵

# Bilinear 神經層計算: 未訓練就是矩陣內積
# 維度(128,20)@(20,40)+(128,30)@(30,40) = (128,40)
output = layer2(input1, input2)
print(output.shape) # torch.Size(128, 40)


'''
3. 引進完全連接層 估算簡單線性回歸參數w, b
'''
import numpy as np

# 產生線性隨機資料100筆，介於0~50
n = 100
X = np.linspace(0, 50, n)
y = np.linspace(0, 50, n)

# 資料加入noise
X += np.random.uniform(-10, 10, n) # 100筆 -10~10 (機率相同)
y += np.random.uniform(-10, 10, n)

# 定義模型: 導入神經網路模型
# 簡單的順序模型(Sequential)內含完全(Linear)神經層與扁平層(flatten)
# 扁平層Flatten: 參數設定要將那些維度轉成一維，起始設定為(0, -1)將所有資料轉成一維
# 0: 指定從哪一維度開始壓平，預設為0=把全部壓平
# -1 指示壓到哪一個維度，-1即是壓到最後一項(全部壓成一維)
# 批次維度（Batch Dimension）是指在一次訓練過程中同時處理的數據樣本數量。
# 順序模型(Sequential): 以序列的方式疊加多個層來構建模型，每一層接收前一層的輸出作為其輸入。

def create_model(input_feature, output_feature): # 輸入特徵數量, 輸出特徵數量
    model = torch.nn.Sequential( # 創建一個順序模型
        torch.nn.Linear(input_feature, output_feature), # 順序1: Linear神經層(輸入特徵數量, 輸出特徵數量)
        torch.nn.Flatten(0, -1) # 扁平層: 把全部壓成一維(全部乘起來)
    )
    return model

# 測試扁平層(Flatten)
input = torch.randn(32, 1, 5, 5) # 輸入資料 = (32, 1, 5, 5)隨機抽樣自常態分布
# 創造一個m = 順序模型
m = torch.nn.Sequential(  
# 二維卷積層(2D Convolutional Layer): 是一種專門用於處理具有二維結構數據的層。常用在圖像識別、物體檢測、圖像分類..
# 有一堆不同形狀和大小的圖章，而這些圖章就是「過濾器」或「卷積核」。
# 而任務是用這些圖章在一張大紙（圖像）上逐一蓋章，每蓋一次就檢查圖章下面的圖畫與圖章是否相似。
# 過濾器/卷積核: 用來在圖像上尋找特定特徵的工具
# 卷積操作： 將圖章從紙張的一角移到另一角，每次移動固定的距離，並在每個位置檢查下方的圖像與圖章的相似度。
# 輸出特徵圖（Feature Maps）：每個過濾器對輸入數據的卷積操作產生一個特徵圖，這個特徵圖反映了輸入數據中特定特徵的空間分佈。
    torch.nn.Conv2d(1, 32, 5, 1, 1), 
# 1: 輸入通道數（Input Channels）,  這個參數決定構成圖像數據的基本顏色成分
# 32: 輸出通道數（Output Channels）, 這個參數決定卷積層將要產生的特徵圖（feature maps）的數量
# 5: 卷積核大小（Kernel Size）: 這個參數定義了每個卷積核的維度(大部分是方陣), 這裡代表5x5方陣
# 1: 步長（Stride）: 代表每次滑動的步長，這裡代表每次移動一像素
# 1: 填充（Padding）: 填充是在輸入數據的邊緣周圍添加的零值，這裡的1代表上下左右都加一層0
    torch.nn.Flatten()
    # torch.nn.Flatten(): 預設(1, -1)，故結果是二維，通常一維代表數據,二維代表分類的預測答案
)
output = m(input) # 32, 32, 3, 3 
print(output.size()) # 32, 288(32*3*3)


'''
4. 定義訓練函數:
神經網路僅使用一個完全連接層, 只有輸入一個神經元(x), 且輸出也只有一個神經元(y)，偏差向(Bias)預設值為True
除了一個神經元的輸出外，還會有一個偏差項，就很像y=wx+b
'''
def train(X, y, epochs=2000, lr=1e-6): # 1e-6: 1*10^-6
    model = create_model(1, 1) # x=1個特徵, y=1個特徵

    # 定義損失函數
    loss_fn = torch.nn.MSELoss(reduction='sum') # reduction='sum': 參數意味著計算所有數據點誤差平方和的總和，而不是平均值。
            # torch.nn.MSELose: 取代MSE公式
# 使用迴圈，反覆進行正向/反向傳導的訓練:
# (1) 計算MSE: 改用loss_fn=(y_pred - y)
# (2) 梯度重置: 改用 model.zero_grad() 取代單獨對w,b的".grad.zero_()"
# (3) 權重更新: 改用 model.parameters 取代對w,b的 w =- lr*w.grad 
# (4) model[0].weight、model[0].bias 可取得權重、偏差
    loss_list, w_list, b_list = [], [], []
    for epoch in range(epochs):
        y_pred = model(X)

        # 計算損失函數
        MSE = loss_fn(y_pred, y) # y_pred, y丟給 torch.nn.MSELose()

        # 梯度重置(改用model.zero_grad())
        model.zero_grad()
        MSE.backward()
        
        # 權重更新
        with torch.no_grad():
            for param in model.parameters():
            # model.parameters()返回所有參數，包括權重w和偏差b。
                param -= lr*param.grad
            # 對所有參數進行更新

        # 紀錄訓練結果
        linear_layer = model[0] # 取得模型(torch.nn.Sequential)的第一層(torch.nn.Linear)
        if (epoch+1)%1000 == 0 or epochs < 1000:
            w_list.append(linear_layer.weight[:, 0].item())
            # :代表選擇所有元素，因此[:, 0]代表選擇所有一維元素, 與二維第一個元素
            b_list.append(linear_layer.bias.item())
            loss_list.append(MSE.item())
            
    return w_list, b_list, loss_list
# 執行訓練
x2, y2 = torch.FloatTensor(X.reshape(X.shape[0], 1)), torch.FloatTensor(y)
# 把x從原本的一維n轉成, (n,1)二維, 二維通常表特徵, 故此為1 
w_list, b_list, loss_list = train(x2, y2, epochs=10**5) # 移除原本書上指定的lr，避免過大問題
# 取得w,b最佳解
print(f'w={w_list[-1]}, b={b_list[-1]}')


'''
5. 以Numpy驗正
'''
coef = np.polyfit(X, y, deg=1)
print(f'w={coef[0]}, b={coef[1]}')


'''
6. 顯示回歸線
'''
import matplotlib.pyplot as plt
plt.scatter(X, y, label='data')
plt.plot(X, w_list[-1] * X + b_list[-1],'r-',label='predict')
plt.legend()
plt.show()


'''
7. Numpy模型繪圖驗證
'''
plt.scatter(X, y, label='data')
plt.plot(X, coef[0] * X + coef[1], 'r-', label='predict')
plt.legend()
plt.show() 


'''
8. 損失函數繪圖
'''
plt.plot(loss_list)
plt.show()
