'''
卷積定義: 一個filter(Kernel), 對圖像進行【乘積和】運算

1. 將圖片依照kernel進行切割成相同size(kernel size)
2. 切下來的圖片與kernel進行相乘(裡面都是0,1)
3. 加總數字, 即為輸出的第一格數值
4. 逐步向右滑動, 去掃完圖片
5. 到最右後, 往下滑
'''

# 1. 準備資料與kernel
import numpy as np

source_map = np.array(list('1110001110001110011001100')).astype(np.int64) # 指定int
source_map = source_map.reshape(5,5) # 5x5矩陣
print('原始資料：') 
print(source_map)

# 過濾器
filter1 = np.array(list('101010101')).astype(np.int64).reshape(3,3) # 改成int64 
print('\n過濾器:') 
print(filter1)


# 2. 計算卷積
width = height = source_map.shape[0] - filter1.shape[0] + 1 
# 當使用大小為 FxF kernel 對 NxN 圖像進行卷積操作時,
# 不考慮填充（padding）和步長（stride）的影響，輸出尺寸O的計算公式為: O=N−F+1

result = np.zeros((width, height)) # 創造輸出圖像大小, 內容都是0的array
# 計算每一格
for i in range(width): 
    for j in range(height):
        value1 =source_map[i:i+filter1.shape[0], j:j+filter1.shape[1]] * filter1 # 給出乘積
        # 從i~i+filter.shape[0] 不包含i+filter.shape[0], filter.shape[0] 第一維大小: 3 
        # 假如 i = 0, 則 0~3 取 0,1,2 
        # 假如 j = 1, 則 1~4 取 1,2,3
        result[i, j] = np.sum(value1) # 計算value1 所有元素的sum 並把這個值給到 result的[i, j]位置中
print('卷積後: \n',result)


# 3. 使用SciPy套件檢查結果
'''
SciPy 是一個開源的Python庫, 用於科學和技術計算
它建立在NumPy庫之上, 提供了一組豐富的數學演算法和函數
涵蓋了數值積分、優化、統計、信號處理、圖像處理、線性代數、傅立葉變換、特殊函數和其他多種科學計算領域 
'''
import scipy 

# convolve2d: 二維卷積
print('SciPy: \n',scipy.signal.convolve2d(source_map, filter1, mode='valid'))
'''
mode: 指定卷積的類型(Padding)
 (1) 'valid': 只計算完全重疊的卷積，這會產生比原始圖像更小的輸出
 (2) 'full': 輸出大小為source_map和 filter1 大小的和-1, 是最完整的卷積模式
 (3) 'same': 輸出大小與source_map相同, 卷積核filter1在source_map中心對齊
'''



'''
卷積計算時，還有兩個參數

 (1) 補零(Padding): 若要補零，可直接指定個數。 
 (2) 步數(Stride): 指Kernel滑動幾格
'''