'''
1. 變數y會自動對變數x進行偏微分
'''
import torch 

# 設定x參與自動偏微分
x = torch.tensor(4.0, requires_grad=True)
# 創造一個張量x 值=4.0
# requires_grad=True: 告訴PyTorch"跟蹤這個張量，並在需要時自動計算它的梯度(對它微分)"

y = x ** 2  # y=x平方

print(y)
print(y.grad_fn) # 紀錄了創造y的操作: 即x平方(梯度函數)
y.backward() # 對y執行反向傳導(偏微x)
print(x.grad) # 張量x對某個損失函數的梯度, 這裡指對y的梯度


'''
2. 取得自動微分的相關屬性
'''
x = torch.tensor(1.0, requires_grad = True)
# x張量=1.0 ,且為被自動微分的對象
y = torch.tensor(2.0) # y張量=2.0
z = x*y # z張量 = x*y

# 顯示自動微分相關屬性
for i, name in zip([x, y, z], 'xyz'): 
# 利用zip(): 將張量列表[x, y, z]和字串'xyz'打包成一個迭代器，
# 每次迭代會提供一個張量(i)和一個對應的名稱(name)。
    print(f'{name}\ndata: {i.data}\nrequires_grad: {i.requires_grad}\n' + 
        'grad: {i.grad}\ngrad_fn: {i.grad_fn}\nis_leaf: {i.is_leaf}\n')
# f{} 運行時會自動替換這些{}的內容為對應變量的值
# 例如: {name}代表name變量的值將被插入到這個位置。
# 如果name變量的值是'x'，那麼{name}就會被替換成'x'

# data: {i.data} 代表顯示出data:並替代i(及[x, y, z]的資料)
# requires_grad: {i.requires_grad}顯示pytorch是否對其自動微分([x, y, z]) [True、False]
# grad: {i.grad}顯示張量的梯度
# grad_fn: {i.grad_fn} 顯示梯度函數
# is_leaf: {i.is_leaf} 顯示是否為"Leaf Node"(葉節點) 
# 葉節點: 由用戶創建的張量，而不是透過其他張量運算而成(例如用torch.tensor就是)


'''
3. 用神經網路時，常用交叉熵(cross entropy)作為損失函數
'''
x = torch.ones(5) # 長度為5的一維張量，裡面都是1
y = torch.zeros(3) # 長度為3的一維張量，裡面都是0
w = torch.randn(5, 3, requires_grad=True)
# w = (5,3)的二維張量，且其中元素都是隨機生成 
# 元素值按照標準正態分佈（平均值為0，標準差為1）
# 最後指定w為自動微分對象
b = torch.randn(3, requires_grad=True) 
# b = 長度3的一維張量， 且其中元素都是按照標準正態分佈隨機生成
z = torch.matmul(x, w)+b
# torch.matmul(x, w): 計算x和w的矩陣乘法 [5]*[5,3] = [3]
# 再跟b相加，而b也是[3]，結果即是z = wx + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# torch.nn.functional.binary_cross_entropy_with_logits 
# PyTorch中的一個函数，用於計算二元交叉熵，適用於二元問題。
# z: 是模型原始輸出
# y: 是目標值(應為0 or 1，已表示二元中的類別)
print('z 梯度函數:', z.grad_fn)
print('loss 梯度函數:', loss.grad_fn)


'''
4. 自動微分
 (1) z是w、b的函數
 (2) loss是z的函數

故只要對loss進行反向傳導即可
'''
loss.backward() # backward(): 拿所有requires_grad=True的張量對其偏微
print(w.grad) # w的梯度(對requires_grad=True對象微分後的值)
print(b.grad) # b的梯度


'''
5. 在PyTorch中，Variable已被棄用，直接使用tensor即可
'''
# from torch.autigrad import Variable 
# x = Variable(torch.one(1), requires_grad=True) 把Variable改成tensor即可
x = torch.ones(1, requires_grad=True)
y = x + 1
y.backward() # 準備拿requires_grad=True的值來微分
print(x.grad) # 1 (y的x偏微結果)


'''
6. 模型訓練後，會反覆執行正向/反向傳導(找最佳解)。因此梯度下降會進行很多次，須注意:
 (1) .backward()執行後，預設會將運算圖銷毀，要保留運算圖: retain_graph=True
 (2) 梯度會不斷累加，因此執行.backward後要reset梯度: .grad.zero_()
'''

'''
7. 不reset梯度(瘋狂累加)
'''
x = torch.tensor(5.0, requires_grad=True) # x=5的張量，自動微分對象
y = x ** 3
y.backward(retain_graph=True) # y進行反向傳導，且保留運算圖(才能對同組數據進行多次下降法)
# 調用.backward()時，PyTorch會根據這個計算圖來計算梯度。
# 為了節省記憶體，計算完梯度後，PyTorch會把這個計算圖銷毀。
# 進而損失掉為了計算梯度，裡面的中間變量們，而不知道要算什麼。
print(f'第一次梯度下降={x.grad}') # 3x^2, x=5, 3*25=75

y.backward(retain_graph=True)
print(f'第二次梯度下降={x.grad}') # 75+75=150

y.backward(retain_graph=True)
print(f'第三次梯度下降={x.grad}') # 75+75+75=225
# 每次呼叫.backward()，這些梯度會自動累積到張量的.grad屬性中。
# 如果沒有重置梯度下，多次執行.backward()，每次計算得到的梯度就會加到前一次的梯度上。


'''
8. 梯度重置(reset): .grad.zero_() 
'''
x = torch.tensor(5.0, requires_grad=True)
y = x ** 3
y.backward(retain_graph=True)
print(f'第一次梯度下降:{x.grad}') # 75
x.grad.zero_() # 計算完後，對x的梯度紀錄reset(重置為0)

y.backward(retain_graph=True)
print(f'第二次梯度下降={x.grad}') # 75
x.grad.zero_()

y.backward(retain_graph=True) 
print(f'第三次梯度下降={x.grad}') # 75


'''
9. 多變數的梯度下降
'''
x = torch.tensor(5.0, requires_grad=True)
y = x ** 3
z = y ** 2 
# 正向: x -> y -> z 
# 故對z往前推

z.backward(retain_graph=True)
print(f'練習:第一次梯度下降={x.grad}') # 18750
x.grad.zero_()

z.backward()
print(f'練習:第二字梯度下降={x.grad}') # 18750


'''
10. 把2-2.neural-networks.py的梯度下降法，改用pytorch做
'''
import numpy as np
import matplotlib.pyplot as plt
#(1)
def Lfunc(x): # 損失函數
    return x ** 2

# 只改動自動微分
def dLfunc(x):
    x = torch.tensor(float(x), requires_grad=True)
    y = x ** 2 # 損失函數
    y.backward()
    return x.grad # 印出損失函數中的x偏微 

def GD(x_start, df, epochs, Ir):
    xs = np.zeros(epochs+1) #(epochs+1)長度列，因為epochs是跌代次數，不包含初始值
    x = x_start
    xs[0] = x # 放入xs第一項
    for i in range(epochs):
        dx = df(x) # 微分
        x = x - Lr*dx # 梯度下降
        xs[i+1] = x # 放入xs中
    return xs # 回傳完成xs

x_start = 5
epochs = 15
Lr = 0.3

w = GD(x_start, dLfunc, epochs, Lr) #function能直接當超參數
print(np.round(w, 2)) # 4捨5入取至小數2位

t = np.arange(-6.0, 6.0, 0.01) # -6~5.99 每隔0.01
plt.plot(t, Lfunc(t), c='b') # 繪製損失函數
plt.plot(w, Lfunc(w), c='r', marker ='o', markersize=5) # 損失函數隨著權重w變化的情況
plt.rcParams['font.sans-serif']=['Microsoft JhengHei'] # 無襯線字體: 微軟正黑
plt.rcParams['axes.unicode_minus']=False # 負號不用unicode
plt.title('梯度下降法', fontsize=20) # 圖表標題
plt.xlabel('X', fontsize=20) # X軸標題
plt.ylabel('損失函數', fontsize=20) # y軸標題
plt.show()

#(2) ==========================================================================
def Lfunc(x): # 損失函數
    return 2*x**4-3*x**2+2*x-20
def dLfunc(x):
    x = torch.tensor(float(x), requires_grad=True)
    y =  2*x**4-3*x**2+2*x-20 # 損失函數
    y.backward()
    return x.grad 
def GD(x_start, df, epochs, Lr):
    xs = np.zeros(epochs+1) 
    x = x_start
    xs[0] = x 
    for i in range(epochs):
        dx = df(x) 
        x = x - Lr*dx 
        xs[i+1] = x 
    return xs 

x_start = 5
epochs = 15000
Lr = 0.001

w = GD(x_start, dLfunc, epochs, Lr) 
print(np.round(w, 2)) 
t = np.arange(-6.0, 6.0, 0.01) 
plt.plot(t, Lfunc(t), c='b') 
plt.plot(w, Lfunc(w), c='g', marker ='*', markersize=5) 
plt.rcParams['font.sans-serif']=['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus']=False 
plt.title('梯度下降法', fontsize=20) 
plt.xlabel('x', fontsize=20) 
plt.ylabel('損失函數', fontsize=20) 
plt.show()


'''
11. 使用梯度下降法，對線性回歸求解。
 (1) 方程式: y = wx + b
 (2) 求w, b 最佳解
'''
# 剛開始的w,b是亂猜的，這裡使用常態分配的隨機亂數
# 損失函數 = MSE
# 權重更新必須在"設定不需要追蹤梯度"才能運算
# 每次訓練後的w,b都需要儲存起來，以便觀察
# 取得w,b的值: 
  #.item()轉為常數
  # .detach().numpy()轉為陣列
  # .detach()作用是脫離梯度下降的控制
# 記得梯度reset
import numpy as np 
import torch
def train(X, y, epochs=100, lr=0.0001):
    loss_list, w_list, b_list=[], [], []# 創造3個空list
# X：訓練用的數據。通常是矩陣，每一行代表不同的數據點，每一列代表不同的特徵。
# y: 每一個數據點對應的目標值，模型的任務是預測這些值。
# epochs: 跌代次數默認為100
# lr: 學習率默認為0.0001
    # w、b 初始值均設為常態分配之隨機亂數 
    w = torch.randn(1, requires_grad=True, dtype=torch.float)#資料型態=浮點數
    # w = 一個常態分配亂數的元素(平均值=0, 標準差=1)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)
    # b = 一個常態分配亂數的元素(平均值=0, 標準差=1)    
    for epoch in range(epochs):   # 跌代次數=epochs       
        y_pred = w * X + b        # y預測值
        
        # 計算損失函數值
        MSE = torch.square(y - y_pred).mean()
        # 損失函數MSE: 計算(預測-目標值)^2，最後取平均        
        MSE.backward()  # 對損失函數進行反向傳導
        
        # 設定不參與梯度下降，w、b才能運算
        with torch.no_grad(): # 設定不參與梯度下降，w,b才能運算
            # 新權重 = 原權重 — 學習率(learning_rate) * 梯度(gradient)
            w -= lr * w.grad
            b -= lr * b.grad 
# 使用已經計算好的梯度（學習過程得到的反饋）來更新模型的權重和偏差。
# 但在這個過程中，不需要也不想PyTorch去記錄這些調整的細節，所以要用with torch.no_grad()         
        # 記錄訓練結果
        if (epoch+1) % 1000 == 0 or epochs < 1000:
            # detach：與運算圖分離，numpy()：轉成陣列
            # w.detach().numpy()
            w_list.append(w.item())  # 把w轉為常數(標準python數字)，並記錄進w_list中
            b_list.append(b.item())  # 把b轉成python表準數字，加進list
            loss_list.append(MSE.item())
      
        # 梯度重置
        w.grad.zero_()
        b.grad.zero_()
        
    return w_list, b_list, loss_list # 回傳紀錄完畢的三個list

# 產生線性隨機資料100筆
n = 100 # 代表要生成100筆
X = np.linspace(0, 50, n)  # np.linspace(): 生成等距數組
y = np.linspace(0, 50, n)  # 生成0~50, 100筆的等距數組
  
# 資料加雜訊
# 雜訊: 給原始數據中加上一些隨機值，模擬現實中數據的不完美性
X += np.random.uniform(-10, 10, n)  # 從均勻分布(uniform)中抽取隨機(random)數
y += np.random.uniform(-10, 10, n)  # 隨機生成區間(-10~10), 生成100個
w_list, b_list, loss_list = train(torch.tensor(X), torch.tensor(y), epochs=100000)

# 取得 w、b 的最佳解
print(f'w={w_list[-1]}, b={b_list[-1]}')


'''
12. 用Numpy 驗證
'''
coef = np.polyfit(X, y, deg=1) 
# np.polyfit():用來找出最佳擬合線來描述一組數據點之間的關係
# deg: 幾次多項式
print(f'w={coef[0]}, b={coef[1]}')


'''
13. 訓練100次的模型繪圖出來
'''
import matplotlib.pyplot as plt
# 真實資料 
plt.scatter(X, y, label='data')
# 散點圖(x軸, y軸, 圖中顯示標籤=data)
# 預測線
plt.plot(X, w_list[-1]*X+b_list[-1], 'r-', label='predicted')
# r-: r(紅色) -(實線), --(虛線), :(點線)
plt.legend() # 使其顯示圖例
plt.show()


'''
14. 用Numpy模型繪圖驗證
'''
plt.scatter(X, y, label='資料')
plt.plot(X, coef[0]*X+coef[1], 'b--', label='預測')
plt.legend()
plt.show()


'''
15. 損失函數繪圖 
'''
plt.plot(loss_list) #MSE
plt.show()
