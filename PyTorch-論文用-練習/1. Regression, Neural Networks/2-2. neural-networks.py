'''
1. 梯度(微分後代入x的數值)下降法 Gradient Descent
'''
# (1) 損失函數可以幫助找到接近目標的新函數
# (2) 數值移動的方向可以從梯度判定
# (3) 損失函數的移動方向與梯度方向相反

import numpy as np
import matplotlib.pyplot as plt 

# 設定損失函數
def Lfunc(x):
    return x**2 #設定損失函數為x的平方

# 損失函數的一階導數(微分)
def dLfunc(x):
    return 2*x  #微分結果為2x

'''
2. 設定進行梯度下降的函數(新x = 目前x - (學習率*梯度)) 
'''
# 根據(2)損失函數的移動方向與梯度方向相反, 所以"減去""(學習率*梯度)
def GD(x_start, df, epochs, Lr):
# x_start: 起始點, 梯度下降算法的x起始點
# df: 一階倒數函數(應放入dLfunc)
# epochs: 迭代次數
# Lr: 學習率, 控制梯度下降過程中的步長大小
    xs = np.zeros(epochs+1) # 初始化(0)長度為epochs+1的陣列, 儲存每一步的x值
    x = x_start # 設定起始點
    xs[0] = x # x初始點放在xs陣列的第一個
    for i in range(epochs): # 進行epochs次迭代
        dx = df(x) # 目前x的微分
        x = x-Lr*dx # 更新x值, 新x=目前x-(學習率*梯度)
        xs[i+1] = x # 新x值存入xs陣列中
    return xs # 跑完迴圈後, 回傳xs陣列

'''
3. 設定起始點、學習率、執行週期後, 呼叫梯度下降法求解
'''
# 超參數(Hyperparameters): 事先給定的, 用來控制學習過程的參數
x_start = 5 #起始點
Lr = 0.3 #學習率
epochs = 15 #執行週期

# 梯度下降法
w = GD(x_start, dLfunc, epochs, Lr=Lr) #可直接按照位置填入, 使用"="則不須照位置
print(np.around(w, 2)) #np.around(w, 2): 對w進行四捨五入, 取到小數第二位 

t = np.arange(-6.0, 6.0, 0.01)
# 創建一個陣列從-6.0~6.0(不包括6.0本人), 數值間隔為0.01, 即從-6.0~5.99每個數值差0.01

# 繪製損失函數本人
plt.plot(t, Lfunc(t), c='b')
# x軸=t, y軸=Lfunc(t), c:顏色='b'藍色

# 繪製梯度下降找最小值的過程
plt.plot(w, Lfunc(w), c='r', marker='o', markersize=5)
# x軸=w, y軸=Lfunc(w), c:顏色='r'紅色, marker:標記點='o'表示為圓形, markersize:標記點大小=5

# 設定字型
# plt.rcParams: 儲存了所有matplotlib的參數, 通常使用類似字典的索引方式
plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
# font.sans-serif: 無襯線字體（Sans-serif fonts）, 這是一字體的總稱, 常見的有Arial、Helvetica、Verdana 和微軟正黑體等
# 指定每當使用"無襯線字體"時, 都使用Microsoft JhengHei(微軟正黑體)
plt.rcParams['axes.unicode_minus']=False
# 表: 是否使用unicode來表示'負號', 因用unicode表示負號可能會有顯示錯誤, 於是不用(False)unicode, 意味著用"減法"表示
plt.title('梯度下降法', fontsize=20) # 標題, 設定字體大小
plt.xlabel('X', fontsize=20) # X軸標題, 字體大小
plt.ylabel('損失函數', fontsize=20) # y軸標題, 字體大小
plt.show() # 開啟圖表

'''
4. 設定損失函數為F(x)=2x^4-3x^2+2x-20, 使用梯度下降法找出最佳解
'''
# 定義損失函數
def Lfunc(x):
    return 2*x**4-3*x**2+2*x-20

# 損失函數微分
def dLfunc(x):
    return 8*x**3-6*x+2

# 繪製損失函數本人
t = np.arange(-6.0, 6.0, 0.01) 
plt.plot(t, Lfunc(t), c='b')
plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus']=False
plt.title('梯度下降法', fontsize=20)
plt.xlabel('x', fontsize=20)
plt.ylabel('損失函數', fontsize=20)

# 梯度下降法
# w = GD(x_start, dLfunc, epochs, Lr=Lr) 
# plt.plot(w, Lfunc(w), c='r', marker='o', markersize=5)
# 原參數會造成錯誤, 原因是學習率太大, 導致錯過最小值後, 一直往左邊跑
''' OverflowError: (34, 'Result too large') '''

# 修改學習率參數、執行次數參數
x_start = 5
Lr = 0.001 # 避免移動幅度過大
epochs = 15000 # 避免還未逼近就結束運算
w = GD(x_start, dLfunc, epochs, Lr=Lr) 
print(np.around(w, 2))

plt.plot(w, dLfunc(w), c='r', marker='o', markersize=5)
plt.show()
