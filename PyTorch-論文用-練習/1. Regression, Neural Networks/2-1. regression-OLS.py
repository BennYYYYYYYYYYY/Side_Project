'''
迴歸 y=wx+b  w權重(斜率)、b誤差(截距)、y依賴變量(目標值)、x獨立變量(特徵)
以普通最小平方法計算'人口統計資料'(MSE最小化)
x = 年度(year), y = 人口(pop)
'''
import matplotlib.pyplot as plt # 視覺化模組
import numpy as np # 陣列計算模組
import math # 數學運算、函數模組
import pandas as pd # 資料處理模組

'''
1. 載入資料 (用pandas)
'''
data = pd.read_csv('C:\\Users\\user\\Desktop\\Python\\PyTorch\\data\\population.csv') # 載入csv檔案
# 計算斜率(w)
w = ((data['pop']-data['pop'].mean())*data['year']).sum()/((data['year']-data['year'].mean())**2).sum()
# p2-7:[y-y平均)*x]加總/[(x-x平均)平方]加總

# 計算截距(b) (計算公式)
b = data['pop'].mean() - w*data['year'].mean() # p2-7: y平均-(斜率w*x平均)
print(f'w={w}, b={b}')
# F-string  f{輸入的值},會將結果轉為字符串


'''
2. 用Numpy中的polyfit函數做,(多項式擬合Polynomial Fitting))
'''
# np.polyfit():用來找出最佳擬合線來描述一組數據點之間的關係
coefficient = np.polyfit(data['year'], data['pop'], deg=1) # deg:幾次多項式, deg=2(二次多項式)
print(f'w={coefficient[0]}, b={coefficient[1]}')
# coefficient=[斜率(w), 截距(b)]


'''
3. 以矩陣表示,將b視為w的一環 (y預測=wx)
'''
# 從data中把'year'選取,並利用.values 把pandas數據結構 轉換成Numpy數列
x = data[['year']].values # 雙[]代表資料為二維DataFrame, 而不是單[]的一維資料Series

# b = b*1
one = np.ones((len(data), 1)) # 創造一個(data有多少row, 1(column))的二維陣列,裡面數據都是1
# 括號1:ones函數、括號2:指定維度、括號3:len

# 合併x與one, 目的是在模型中引入截距項
x = np.concatenate((x, one), axis=1) # np.concatenate((指定要合併的), axis=指定軸:0=column,1=row)
# print('check', x)

y = data[['pop']].values # data中提取'pop'DataFrame轉換成numpy數值

# 正規方程式(Normal Equation): w=[(轉置x)乘(原x)的逆矩陣]乘(轉置x)乘y(目標值)
w = np.linalg.inv(x.T@x)@x.T@y
# np.linalg.inv():逆矩陣函式(linear algebra inverse), @:矩陣乘法, .T:轉置矩陣函式

print(f'w={[0, 0]}, b={[1, 0]}') # print出來只是w=[0, 0]與b=[1, 0]
print('可能才是正確的?:', f'w={w}, b={b}')

'''
4. 以Scikit-Learn房價資料集為例, 求解線性回歸 (資料集load_bostom已被移除, 改用california_housing)
'''
from sklearn.datasets import fetch_california_housing  # Scikit-learn為機器學習庫, 並從中下載數據集


# 把特徵變數給變量x, 把目標變量給變量y
# california_housing = fetch_california_housing()
# x = california_housing.data
# y = california_housing.target
X, y = fetch_california_housing(return_X_y=True) 
# return_X_y=True: 返回特徵矩陣x與目標向量y，而不是字典，方便直接使用數據做分析/訓練

one = np.ones((X.shape[0], 1))
# 建立都是1的二維陣列 (x資料行(column)數量, 1)

x = np.concatenate((X, one), axis=1)
# x與one合併, 指定axis=row(2維)

w = np.linalg.inv(X.T@x)@X.T@y
# Normal Equation


'''
5. 以Scikit-Learn 的線性回歸類別驗證答案
'''
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
# 下載線性回歸模型

# 下載資料
# california_housing = fetch_california_housing()
# x = california_housing.data
# y = california_housing.target
X, y = fetch_california_housing(return_X_y=True) 

Ir = LinearRegression() # 創造線性回歸的實體物件, 命名為lr
Ir.fit(X, y) #Ir物件用,(X, y)去fit

# 取得模型參數
Ir.coef_, Ir.intercept
# 參數, 截距

'''
6. PyTorch直接呼叫線性代數函數庫
'''
import torch
from sklearn.datasets import fetch_california_housing

# 下載資料
X, y = fetch_california_housing(return_X_y=True)

X_tensor = torch.from_numpy(X) # 把Numpy中的數組轉為Tensor
# 在PyTorch, 數據結構是Tensor, 類似numpy中的ndarray

one = torch.ones((X.shape[0], 1))
# 建立二維Tensor(X列數量, 1)

X = torch.cat((X_tensor, one), axis=1)
# X_tensor與one concatenate起來(列)

w = torch.linalg.inv(X.T@X)@X.T@y
print(w)






