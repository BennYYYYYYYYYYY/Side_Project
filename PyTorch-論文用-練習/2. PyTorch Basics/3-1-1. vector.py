# vector(向量):指的是一維的張量，能表長度與方向
'''
1. 長度(magnitude): 計算公式為毆幾米德距離(Euclidean distance)  
'''
# 向量vector
import numpy as np
v = np.array([2, 1]) # 建立一維陣列

# 向量長度計算(magnitude)
magnitude = (v[0]**2+v[1]**2)**(1/2) # (v陣列第一個數平方 + v陣列第二個數平方)開根號
print('1:', magnitude)

# 可以使用 np.linalg.norm()計算 (linear algebra norm)
magnitude = np.linalg.norm(v)
print('2:', magnitude) # 答案一樣

# 也可以使用 pytorch 中的linalg.norm
import torch
magnitude = torch.linalg.norm(torch.FloatTensor(v)) # 需要先把ndarray轉成浮點數張量（tensor）
print('3:', magnitude) # 答案一樣

'''
2. 長度(direction): 使用atan()
'''
import math 

# 向量 三角函數中使用的是弧度(radian)，可轉換成一般較常用的角度(degree)
v = np.array([2, 1])
vTan = v[1]/v[0] # 1/2 
print('tan(θ) = 1/2')
theta = math.atan(vTan) # 使用atan()去算出θ角度
print('弧度radian:', round(theta,4)) # 四捨五入到小數第五位
print('角度degree:', round(theta*180/math.pi,2)) # 換算degree單位, 取道小數第二位
# 也可以使用math.degree() 轉換角度單位
print('角度:', round(math.degrees(theta), 2))

'''
3. 向量加減法
'''
# 1. 對常數: 直接對每個元素進行運算
# 2. 對另一個向量: 需對應相同位置的向量進行運算
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 限制 OpenMP 使用單線程
import matplotlib.pyplot as plt 

# 向量+2
v = np.array([2, 1])
v1 = np.array([2, 1])+2
v2 = np.array([2, 1])-2

# 原點
origin = [0], [0] # 建立一個元組, 裡面有兩個列表, 只有0

# 畫有箭頭的線
plt.quiver(*origin, *v1, scale=10, color='r')
# plt.quiver(): 畫箭頭函數, *指把元組解包成單獨的參數, scale: 控制箭頭比例, color: 指定顏色
# or plt.quiver(origin[0][0], origin[1][0], v1[0], v1[1], scale=10, color='r')
plt.quiver(*origin, *v, scale=10, color= 'b') # 藍色
plt.quiver(*origin, *v2, scale=10, color='g') # 綠色

plt.annotate('origin vector', (0.025, 0.01), xycoords='data', fontsize=16)
# plt.annotate('要註解的文字', (註解文字的座標)), xycoords: 指定註解坐標系的類型, ='data':基於數據坐標系
# 關於坐標系: 可能有不同的類型，例如數據'data', 像素坐標系'figure pixels'ㄝ, 分數坐標系 'axes fraction'
# 因此座標指定完以後，也要記得xycoords指定坐標系

# 作圖
plt.axis('equal') # 調整圖形的座標比例,'equal': 使他x與y軸比例相等
plt.grid() # 添加網格線

plt.xticks(np.arange(-0.05, 0.06, 0.01), labels=np.arange(-5, 6, 1))
# plt.xticks(x軸上實際生成的刻度們, label:給觀眾看到的標籤本人 ): 自定義x軸刻度 (此案例中:將圖放大100倍)
plt.yticks(np.arange(-3, 5, 1)/100, labels=np.arange(-3, 5, 1)) # 放大100倍
plt.show()

'''
4. 向量乘除法: 乘除常數，"長度"改變，"方向"不變
'''
v = np.array([2, 1])
v1 = np.array([2, 1]) *2
v2 = np.array([2, 1]) /2
origin = [0], [0] #原點
plt.quiver(*origin, *v1, scale=10, color='r') # 畫原點到v1的箭頭
plt.quiver(*origin, *v, scale=10, color='b') # 畫原點到v的箭頭
plt.quiver(*origin, *v2, scale=10, color='g') # 畫原點到v2的箭頭
plt.annotate('origin vector', (0.025, 0.008), xycoords='data', color='b', fontsize=16) #加上註解
plt.axis='equal' #xy軸對稱
plt.grid() # 加入網格
plt.xticks(np.arange(-0.05, 0.06, 0.01), labels=np.arange(5,-6,1)) #把x軸放大100倍
plt.yticks(np.arange(-3, 5, 1)/100, labels=np.arange(-3, 5, 1)) #把y軸放大100倍
plt.show()

'''
5. 向量加減乘除另一"向量": 同位置的元素做運算
'''
v = np.array([2, 1])
s = np.array([-3, 2])
v1 = v+s
origin=[0], [0] #原點
plt.quiver(*origin, *v, scale=10, color='r')
plt.quiver(*origin, *s, scale=10, color='b')
plt.quiver(*origin, *v1, scale=10, color='g')
plt.annotate('origin vector',(0.025, 0.008), xycoords='data', fontsize=16, color='b')
plt.axis='equal'
plt.grid()
plt.xticks(np.arange(-0.05, 0.06, 0.01), np.arange(-5, 6, 1))
plt.yticks(np.arange(-3, 5, 1)/100, np.arange(-3, 5, 1))
plt.show()

'''
6. 內積: 點積(dot product) 
(一). 一維座標: v@s=v1*s1 + v2*s2 + v3*s3 ....  
(二). 兩向量長度乘積*cos(θ)

點積可以看兩個向量之間的關係。(投影長*投影長)
 (1) 點積是正數: 兩向量指向同一方向
 (2) 點積是負數: 指向相反方向
 (3) 點積為零: 夾角90度
'''
v = np.array([2, 1])
s = np.array([-3, 2])
d = v@s # @:內積運算 
print(d) # -4 = 2*(-3) + 1*2

'''
7. 計算向量夾角 
cos(θ): 兩個向量的方向關係
 (1) θ=0,cos(θ)=1: 兩向量同方向
 (2) θ=90,cos(θ)=0: 兩向量反方向
'''
v = np.array([2, 1])
s =np.array([-3, 2])
vMag=np.linalg.norm(v) # 計算v的magnitude(長度)
sMag=np.linalg.norm(s) # 計算s的長度
cos = (v@s)/(vMag*sMag) # 算出cos(θ)
theta = math.degrees(math.acos(cos)) # 為求θ，需乘上acos抵銷cos，再將弧度轉角度
# math.degrees(x)：弧度x轉角度，math.acos(x)：對x取acos
print(theta)




