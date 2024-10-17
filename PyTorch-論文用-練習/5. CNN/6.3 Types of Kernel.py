# 雖然CNN會自動配置卷積種類，但還是看看各種效果。
'''
Python OpenCV套件: 是一個影像處理的套件(pip install opencv-python)

skimage: 全名 scikit-image, 也是一個影像處理套件, 功能較OpenCV簡易
'''
import numpy as np
from skimage.exposure import rescale_intensity
from scipy.signal import convolve2d
import skimage
import cv2 # pip install opencv-python

source_map = np.array(list('1110001110001110011001100')).astype(np.int64) # 指定int
source_map = source_map.reshape(5,5) # 5x5矩陣
print('原始資料：') 
print(source_map)

# 過濾器
filter1 = np.array(list('101010101')).astype(np.int64).reshape(3,3) # 改成int64 
print('\n過濾器:') 
print(filter1)






# 1. 計算卷積

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





# 2. 卷積影像轉換函數

def convolve(image, kernel):
    # 取得圖像與濾波器的寬高
    (iH, iW) = image.shape[:2] # 0與1維大小
    (kH, kW) = kernel.shape[:2]

    # 計算 padding='same' 單邊所需的補零行數
    pad = int((kW - 1) / 2) # ((kW - 1) / 2): 計算padding的公式, 會給出【需要在每邊加多少像素】的數值
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # 使用OpenCV函數copyMakeBorder對圖像進行邊緣填充
    # pad: 上、下、左、右四個方向的填充
    # cv2.BORDER_REPLICATE: 表示用邊緣像素的值來重複填充新的邊緣區域，即將最邊緣的像素值向外擴展
    output = np.zeros((iH, iW), dtype="float32") # 形狀為(iH, iW)的全零矩陣output

    # 卷積
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):    
        # 從原始圖像的邊緣開始進行卷積計算，所以需要跳過邊緣填充的部分，從 pad 開始
        # 希望能夠覆蓋到原始圖像的所有部分，包括最後一行，但不包括填充的邊緣。因此，需要遍歷到iH + pad的位置。        
        
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]  # Region of Interest, ROI
            # 利用y和x作為當前的中心點，向周圍擴展以選取一個特定大小的區塊
                     
            k = (roi * kernel).sum() # 局部區域與Kernel進行點乘積, 最後加總                   
            output[y - pad, x - pad] = k # 更新計算結果k的值, 到全零矩陣中的相對位置中 

    # 調整影像色彩深淺範圍至 (0, 255)
    output = rescale_intensity(output, in_range=(0, 255)) # rescale_intensity: 用於調整圖像的強度範圍
    # 對圖像的像素強度進行重新縮放，以使得處理後的圖像像素值落在特定的範圍內 in_range(0, 255)

    output = (output * 255).astype("uint8") # *255 是基於data範圍在[0,1]
    # 將結果數組的數據類型轉換成無符號8位整數（Unsigned 8-bit Integer, uint8）

    return output     # 回傳結果影像




# 3. 將影像灰階化

# 自 skimage 取得內建的圖像
image = skimage.data.chelsea() # 取得skimage中的範例數據集 'Chelsea猫'
cv2.imshow("original", image) # 秀出貓貓圖

# 灰階化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# cv2.cvtColor 是OpenCV中用於轉換圖像顏色空間的函數。它可以將圖像從一種顏色格式轉換成另一種
# cv2.COLOR_BGR2GRAY 是指定轉換類型的參數，表示將圖像從BGR顏色空間轉換到灰度空間
'''
需要注意: 
OpenCV預設讀取圖像的顏色順序是BGR, 而不是常見的RGB。
這意味著是直接使用OpenCV讀取圖像的話, 圖像數據是按藍綠紅的順序排列的。
然而, 如果是從skimage或其他庫獲取的RGB格式圖像, 則可能需要先將其轉換為BGR格式再進行灰度轉換。

不過, 直接將RGB圖像用在這個函數中, 通常也能得到正確的灰度圖像
因為轉換成灰度圖像主要是基於亮度而不是顏色通道的具體排序。
'''
cv2.imshow("gray", gray) # 秀出灰色貓貓

# 按 Enter 關閉視窗
cv2.waitKey(0)
# cv2.waitKey() 是一個鍵盤綁定函數，等待任何鍵盤事件
# cv2.waitKey(0)，這表示函數會無限期地等待鍵盤事件

cv2.destroyAllWindows()
# cv2.destroyAllWindows() 會關閉所有OpenCV創建的窗口





# 4. 將影像模糊化(Blur)

# 小模糊 kernel
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
# 創造7x7, 數值為1.0的陣列, 且 *1/49後, 全部的數值加起來會=1
'''
當卷積核的所有值之和為1時, 這意味著在卷積操作中 ,每個像素及其周圍像素的值被平均，但總體亮度保持不變

當卷積核的值之和正好為1時, 卷積後的圖像才會保持與原始圖像相同的亮度水平
因為這相當於對像素值進行了平均分配，而沒有額外增加或減少亮度。
'''
# 卷積
convoleOutput = convolve(gray, smallBlur) # 使用模糊化的kernel
opencvOutput = cv2.filter2D(gray, -1, smallBlur)
# cv2.filter2D 是OpenCV中用於執行二維過濾的函數。這個函式可以用於實現多種圖像處理效果，包括模糊、銳化等
# -1 指定了輸出圖像的深度（即數據類型）與輸入圖像相同 (-1表示輸出圖像將與源圖像有相同的數據類型), 這邊是指8-bit (0~255)
cv2.imshow("little Blur", convoleOutput)


# 大模糊kernel (更模糊), 平均化時考慮的範圍更廣
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
# 卷積
convoleOutput = convolve(gray, largeBlur)
opencvOutput = cv2.filter2D(gray, -1, largeBlur)
cv2.imshow("large Blur", convoleOutput)

# 按 Enter 關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()




# 銳化(sharpen): 可以使圖像的對比更明顯
'''
銳化(Sharpening) 是一種增強數位影像清晰度的圖像處理技術。
目的在於使圖像的細節更為顯著，尤其是邊緣區域。
銳化工作通過增強圖像中顏色或亮度的變化率來達成，這種變化率通常在物體的邊界附近最為顯著。
'''
# sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int") 

# 卷積
convoleOutput = convolve(gray, sharpen)
opencvOutput = cv2.filter2D(gray, -1, sharpen)
cv2.imshow("sharpen", convoleOutput)

cv2.waitKey(0)
cv2.destroyAllWindows()




# Laplacian 邊緣偵測: 可以偵測圖形的輪廓
'''
Laplacian 邊緣檢測: 是一種在數位影像處理中常用的二階導數方法，用於捕捉影像中的邊緣資訊。
這種方法基於拉普拉斯算子(Laplacian operator)，一個數學上用於衡量函數的二階導數，
表示圖像亮度變化的速率的變化速率。Laplacian 邊緣檢測特別適合於揭示圖像細節和偵測邊緣，但它對噪聲也相當敏感。
'''
# Laplacian filter
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int") # 此kernel為固定的, 要用 Laplacian邊緣偵測 就是用這個kernel

# 卷積
convoleOutput = convolve(gray, laplacian)
opencvOutput = cv2.filter2D(gray, -1, laplacian)
cv2.imshow("laplacian edge detection", convoleOutput)

# 按 Enter 關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows() 




# Sobel X軸邊緣檢測: 沿著x軸偵測邊緣, 故可偵測垂直線特徵 
# 與拉普拉斯算子類似，但對噪音容忍度更高，且可偵測邊緣之方向性
'''
Sobel算子: 使用兩個3x3的Kernel, 分別計算
 (1) 水平方向
 (2) 垂直方向 的亮度變化率。

這兩個方向上的變化率合在一起就形成了圖像的邊緣資訊。
'''

# Sobel x-axis filter
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int") # x軸 kernel 的固定樣子

# 卷積
convoleOutput = convolve(gray, sobelX)
opencvOutput = cv2.filter2D(gray, -1, sobelX)
cv2.imshow("x-axis edge detection", convoleOutput)

# 按 Enter 關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()


# Sobel y-axis filter
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int") # y軸 kernel 的固定樣子

# 卷積
convoleOutput = convolve(gray, sobelY)
opencvOutput = cv2.filter2D(gray, -1, sobelY)
cv2.imshow("y-axis edge detection", convoleOutput)

# 按 Enter 關閉視窗
cv2.waitKey(0)
cv2.destroyAllWindows()