'''
參考文章: https://medium.com/lifes-a-struggle/hog-svm-c2fb01304c0

Histogram of Oriented Gradients (HOG) 特徵和支持向量機 (SVM) 來進行影像分類

HOG 的全名是 Histograms of Oriented Gradients 是一種特徵提取的技術，
透過區塊中 Gradient 方向來分別統計累積的 Gradient 強度，在影像上計算 Gradient 能夠凸顯影像的輪廓與邊緣，
經由這些資訊我們就能了解影像當中的物體是什麼。平滑區域因為沒有輪廓資訊 Gradient 計算出來也會很小。
並以此作為該區塊的特徵。

再來就從這些 Gradient 提取出我們想要的特徵，一般來說會分成區塊來作分析，
比較能夠避免受到單一 pixel 數值的影響，也能很有效的降低雜訊干擾。

至於為何選擇 1 個 cell 是 8 x 8 個 pixels? 其實只是實驗過後發現結果較好。

首先將 Gradient 角度分類成 0, 20, 40, 60, …, 140, 160 這 9 個類別組成的特徵向量
選擇好要分成 0–180 之後 (其中 0 跟 180 因為只差在方向，所以用 0 代替)，
現在就只要將每個 pixel 的 Gradient 強度填入特徵向量中對應的 Gradient 方向即可

將每一個 cell (8 x 8 pixels) 計算完後，會得到各個 Gradient 方向(9個方向)的 Gradient 強度加總。

由於 Gradient 的計算很容易受到影像的明亮度影響，
影像越亮 Gradient 強度也就越大，因此需要作 Normalization 降低影像明亮度的影響。
不過不是只對一個 cell(9個向量) 作 Normalization，而是由 4 個 cell 組成的 block 一起作 Normalization，
也就是說一次會有 4 個特徵向量一起作 Normalization ，得出 36(9個向量*4個Cell) x 1 的特徵向量代表 block。

把每一個 block 都 Normalize 成 36 x 1 的特徵向量後，再將這些特徵向量全部接在一起變成一個很大的特徵向量來代表這個 ROI。以 ROI 大小為 64 x 128 為例，
總共可以分成 8 x 16 個 cell，會有 7 x 15 個 block，因此就會產生 7 * 15 * 36 = 3780 維度的特徵向量。
(Block 雖然是 2x2的cell 但每次移動只會移動一格cell，也就是說，會有重複的)

有了特徵向量後，就將特徵向量輸入 SVM 作訓練，最終訓練出能夠正確判斷行人的 SVM model。

SVM 的全名是 Support Vector Machine ，是一種機器學習的技術。
簡單來說， SVM 是希望能找出超平面 (hyperplane) 正確區分不同類別的資料
'''
# Scikit-Image 的範例
# 載入套件
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog # hog 函數用來計算 Histogram of Oriented Gradients 特徵的
from skimage import data, exposure, color, transform # data 用來載入測試圖片的模組, exposure 用來調整圖像對比度的模組

# 測試圖片
image = data.astronaut() # 載入一張內建的測試圖片
image = color.rgb2gray(image) # 新版本中沒有multichannel，所以我改用灰色

# 計算圖片的 HOG 特徵
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True) # 這裡原本有用彩色的語法
# orientations=8 指定使用8個方向
# pixels_per_cell=(16, 16) 指定每個單元格的大小為16x16像素
# cells_per_block=(1, 1) 指定每個塊包含1x1個單元格
# visualize=True 指定返回一個可視化的 HOG 圖像
# multichannel=True 指定處理多通道(彩色)圖像
# fd 是特徵描述符，hog_image 是 HOG 圖像

# 原圖與 hog圖比較 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
# plt.subplots(1, 2) 創建一個包含一行兩列的子圖
# figsize=(12, 6) 指定圖形的大小
# sharex=True, sharey=True 指定子圖共享 X 和 Y 軸

ax1.axis('off') # 關閉軸顯示
ax1.imshow(image, cmap=plt.cm.gray) # 顯示原始圖像，使用灰度顏色地圖
ax1.set_title('Input image') 

# 調整對比，讓顯示比較清楚
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
# exposure.rescale_intensity: 用於重新調整圖像的強度值(對比度)
# in_range=(0, 10): 指定輸入圖像的強度值範圍

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# 收集正樣本 (positive set)
# 使用 scikit-learn 的人臉資料集
from sklearn.datasets import fetch_lfw_people # 載入 Labeled Faces in the Wild 人臉資料集
faces = fetch_lfw_people() 
positive_patches = faces.images # # 提取資料集中的圖像部分作為正樣本
positive_patches.shape # 獲取正樣本圖像的形狀

# 顯示正樣本部份圖片
fig, ax = plt.subplots(4,6) # 創建一個 4 行 6 列子圖
for i, axi in enumerate(ax.flat): # ax.flat 展平子圖陣列，以便使用單一迴圈訪問
    axi.imshow(positive_patches[500 * i], cmap='gray') # 顯示正樣本圖像，使用灰度顏色地圖
    # [500 * 1]: 索引，500 * i 表示每隔 500 張圖像選取一張
    # 這樣做的原因是為了在顯示圖像時不顯示連續的圖像，而是顯示資料集中間隔的一些圖像，以便更全面地查看資料集的多樣性
    axi.axis('off')

# 收集負樣本 (negative set)
# 使用 Scikit-Image 的非人臉資料
from skimage import data, transform, color # color：包含顏色空間轉換功能，例如將RGB圖像轉換為灰度圖像

imgs_to_use = ['hubble_deep_field', 'text', 'coins', 'moon',
               'page', 'clock','coffee','chelsea','horse'] # 要加載的圖像名稱列表
images = [color.rgb2gray(getattr(data, name)()) # 遍歷 imgs_to_use 列表中的每個圖像名字
          for name in imgs_to_use] 
# getattr(data, name)() 來取得這個圖像。data 是導入的模塊，name 是圖像名字，getattr 函數用來動態地取得圖像
# color.rgb2gray：將RGB圖像轉換為灰度圖像
# images：包含所有轉換為灰度的圖像列表
len(images) # 返回圖像列表的長度

# 將負樣本轉換為不同的尺寸 
# 需要從圖像中提取一些小塊
from sklearn.feature_extraction.image import PatchExtractor # PatchExtractor: 從圖像中提取小塊(patch)的工具

# 轉換為不同的尺寸
def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    # img: 輸入的圖像
    # N: 要提取的小塊數量
    # scale: 縮放比例
    # patch_size: 小塊的尺寸 (預設為 positive_patches[0].shape)

    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int)) 
    # 把 patch_size 轉換成 numpy 數組，乘以縮放比例 scale，然後轉換成整數，最後轉換成元組

    extractor = PatchExtractor(patch_size=extracted_patch_size, 
                               max_patches=N, random_state=0) # random_state=0：設定隨機種子，以確保結果可重複
    patches = extractor.transform(img[np.newaxis]) # extractor: 提取小塊
    # img[np.newaxis] 在數組 img 上增加一個維度 PatchExtractor 的輸入要求
    if scale != 1: # 檢查縮放比例是否不等於 1，如果不等於 1，則進行大小調整
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches]) # 使用 transform.resize 方法來調整每個小塊的大小
    return patches

# 產生 27000 筆圖像
negative_patches = np.vstack([extract_patches(im, 1000, scale) # 對每個圖像 im 和每個縮放比例 scale (0.5，1.0，2.0)，提取1000個小塊
                              for im in images for scale in [0.5, 1.0, 2.0]])
# 用 np.vstack 把這些小塊垂直堆疊起來，形成一個大的數組

negative_patches.shape # 顯示 negative_patches 的形狀，確認提取的小塊數量和大小 

# 顯示部份負樣本
fig, ax = plt.subplots(4,6) # 創建一個包含 4 行 6 列子圖的圖表
for i, axi in enumerate(ax.flat): # 迭代每個子圖
    axi.imshow(negative_patches[600 * i], cmap='gray') # 在每個子圖中顯示一個負樣本小塊，使用灰度顏色顯示
    axi.axis('off')

# 合併正樣本與負樣本
from skimage import feature   # skimage.feature 模組，用於使用 hog 函數
from itertools import chain # chain 函數，用於將多個可迭代對象連接起來

X_train = np.array([feature.hog(im) # feature.hog(im): 使用 hog 函數提取影像 im 的 HOG 特徵
                    for im in chain(positive_patches,  
                                    negative_patches)]) # 將正樣本小塊和負樣本小塊連接起來

y_train = np.zeros(X_train.shape[0]) # 創建一個全零的array，長度與 X_train 的一維相同
y_train[:positive_patches.shape[0]] = 1 #  將array 前 positive_patches.shape[0] 個元素設置為 1，表示正樣本。
# 其餘元素保持為 0，表示負樣本。

# 使用 SVM 作二分類的訓練
from sklearn.svm import LinearSVC # sklearn 中的支持向量機 (SVM) 分類器 LinearSVC
from sklearn.model_selection import GridSearchCV # 參數搜索工具 GridSearchCV

'''
LinearSVC 是一種支持向量機(SVM)分類器，它適用於線性可分的數據

參數 C 是一個超參數，
C: 正則化參數。它控制著誤分類數據的懲罰強度。
較小的 C 值表示較強的正則化，這有助於避免過擬合(訓練數據)。較大的 C 值則可能導致過擬合


GridSearchCV 是一個用於超參數調整的工具。其主要參數包括：

estimator: 要調整的模型。在這裡是 LinearSVC(dual=False)。
param_grid: 超參數的選擇範圍。在這裡是一個字典，包含 C 的不同可能值 [1.0, 2.0, 4.0, 8.0]。
cv: 交叉驗證的摺數。在這裡設為3，表示三摺交叉驗證。

交叉驗證 (Cross-validation) 是將數據集分成 k 個摺，訓練 k 次，每次使用 k-1 個摺作為訓練集，
剩餘的一個摺作為驗證集。這樣可以確保模型在不同的訓練/驗證分割下的穩定性
'''
# 使用 GridSearchCV 尋求最佳參數值 (建立 GridSearchCV 物件，進行參數調優)
grid = GridSearchCV(LinearSVC(dual=False), {'C': [1.0, 2.0, 4.0, 8.0]},cv=3) # dual=False 用於樣本數大於特徵數的情況
# LinearSVC(dual=False): 初始化 LinearSVC 模型，並設定 dual=False
# {'C': [1.0, 2.0, 4.0, 8.0]}: 定義一個字典，其中 C 是正則化參數，它的可能值有 1.0, 2.0, 4.0, 8.0
# cv=3: 使用三摺交叉驗證來評估每個參數組合的性能
# 在訓練過程中，GridSearchCV 將遍歷 {'C': [1.0, 2.0, 4.0, 8.0]} 中的每個 C 值，並對每個值進行三摺交叉驗證。對於每個 C 值，它將：
# 1. 將訓練數據分成三個摺。
# 2. 進行三次訓練，每次選擇不同的一摺作為驗證集，其餘兩摺作為訓練集
# 3. 計算每個摺的性能指標（例如準確度），並取三次的平均值作為該 C 值的性能評估
# 最終，GridSearchCV 會選擇性能最佳的 C 值，並將其保存在 grid.best_params_ 中
grid.fit(X_train, y_train)
grid.best_score_ # 顯示最佳參數對應的交叉驗證得分

grid.best_params_ # C 最佳參數值

model = grid.best_estimator_ # 獲取最佳參數對應的模型
model.fit(X_train, y_train) # 使用最佳模型訓練數據

# 取新圖像測試
test_img = data.astronaut() #  載入一張示例圖像
test_img = color.rgb2gray(test_img) # 轉換成灰色
test_img = transform.rescale(test_img, 0.5) # 將圖像縮放到一半大小
test_img = test_img[:120, 60:160] 
# 從起始行到第 119 行的所有行。
# 從第 60 列到第 159 列的所有列


plt.imshow(test_img, cmap='gray')
plt.axis('off')


# 滑動視窗函數
def sliding_window(img, patch_size=positive_patches[0].shape,
                   istep=2, jstep=2, scale=1.0):
    # img：輸入的圖像
    # patch_size：塊的大小，預設 positive_patches[0].shape
    # istep：在行方向上滑動視窗的步長
    # jstep：在列方向上滑動視窗的步長
    # scale：縮放比例，預設1.0，即不縮放
    Ni, Nj = (int(scale * s) for s in patch_size) # 對元組中的每個元素進行縮放並轉換為整數
    # patch_size 是一個包含高度和寬度的元組，表示塊的原始大小
    for i in range(0, img.shape[0] - Ni, istep): # img.shape[0] 和 img.shape[1] 分別是圖像的高度和寬度
        # 從 0 到 img.shape[0] - Ni，以 istep 為步長進行迭代
        for j in range(0, img.shape[1] - Ni, jstep):
            # 從 0 到 img.shape[1] - Nj，以 jstep 為步長進行迭代
            patch = img[i:i + Ni, j:j + Nj] # 表示行和列的範圍，從而在圖像 img 中提取一個大小為 Ni x Nj 的塊
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch # 返回塊的左上角座標 (i, j) 和塊本身 patch

# 使用滑動視窗計算每一視窗的 Hog
indices, patches = zip(*sliding_window(test_img))
# 使用 sliding_window 函數獲取每個視窗的索引和塊，並使用 zip 解壓縮結果，分別存儲在 indices 和 patches 中

patches_hog = np.array([feature.hog(patch) for patch in patches])
# 使用列表生成式對每個塊計算 HOG 特徵，並將結果存儲在 patches_hog 數組中

# 辨識每一視窗
labels = model.predict(patches_hog) # 使用訓練好的模型 model 對 patches_hog 進行預測，返回每個視窗的標籤
labels.sum() # 偵測到的總數

# 將每一個偵測到的視窗顯示出來
fig, ax = plt.subplots()
ax.imshow(test_img, cmap='gray')
ax.axis('off')


# 取得左上角座標
Ni, Nj = positive_patches[0].shape
# Ni, Nj = positive_patches[0].shape：獲取正樣本塊的大小 Ni 和 Nj
indices = np.array(indices) # 轉 numpy 數列


# 顯示
for i, j in indices[labels == 1]: # 遍歷所有標籤為 1 的視窗索引
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))
    # 使用 ax.add_patch 在每個視窗的位置添加一個紅色的矩形框
    
patches_hog.shape
# 返回 patches_hog 數組的形狀，即所有提取的 HOG 特徵的總數及每個特徵向量的大小

candidate_patches = patches_hog[labels == 1] # 篩選出所有標籤為 1 的 HOG 特徵
candidate_patches.shape # 返回候選塊的形狀，即偵測到的候選塊的總數及每個特徵向量的大小

# Non-Maximum Suppression演算法 by Felzenszwalb et al.
# boxes：所有候選的視窗，overlapThresh：視窗重疊的比例門檻
def non_max_suppression_slow(boxes, overlapThresh=0.5):
# 候選視窗的座標列表，每個視窗用一個包含4個元素的列表表示 [x1, y1, x2, y2]
# overlapThresh：重疊門檻值，如果兩個視窗的重疊比例超過這個值，則只保留其中一個
    if len(boxes) == 0:
        return []
    # 檢查 boxes 是否為空。如果是空的，則返回一個空列表
    
    pick = []        # 儲存篩選的結果
    x1 = boxes[:,0]  # 取得候選的視窗的左/上/右/下 座標
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # 計算候選視窗的面積
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)   # 依視窗的底Y座標排序
    
    # 比對重疊比例
    while len(idxs) > 0:
        # 最後一筆
        last = len(idxs) - 1
        # 取得當前 idxs 中的最後一個索引 i，並將其添加到 pick 
        i = idxs[last]
        pick.append(i)
        suppress = [last] # 初始化一個 suppress 列表來記錄
        
        # 比對最後一筆與其他視窗重疊的比例
        for pos in range(0, last):
            j = idxs[pos]
            
            # 取得所有視窗的涵蓋範圍
            xx1 = max(x1[i], x1[j]) # 重疊區域的左上角座標
            yy1 = max(y1[i], y1[j]) 
            xx2 = min(x2[i], x2[j]) # 重疊區域的右下角座標
            yy2 = min(y2[i], y2[j]) 
            w = max(0, xx2 - xx1 + 1) # 重疊區域的寬度和高度
            h = max(0, yy2 - yy1 + 1)
            
            # 計算重疊比例
            overlap = float(w * h) / area[j]
            
            # 如果大於門檻值，則儲存起來
            if overlap > overlapThresh:
                suppress.append(pos)
                
        # 刪除合格的視窗，繼續比對
        idxs = np.delete(idxs, suppress)
        
    # 傳回合格的視窗
    return boxes[pick]

# 使用 Non-Maximum Suppression演算法，剔除多餘的視窗。
candidate_boxes = []
for i, j in indices[labels == 1]:
    candidate_boxes.append([j, i, Nj, Ni])
# 遍歷所有標籤為1的視窗索引，並將其左上角座標 (j, i) 和大小 (Nj, Ni) 添加到 candidate_boxes 中

final_boxes = non_max_suppression_slow(np.array(candidate_boxes).reshape(-1, 4))
# 將 candidate_boxes 轉換為NumPy數組並重塑為 (N, 4) 的形狀，然後應用 non_max_suppression_slow 函數

# 將每一個合格的視窗顯示出來
fig, ax = plt.subplots()
ax.imshow(test_img, cmap='gray')
ax.axis('off')

# 顯示
for i, j, Ni, Nj in final_boxes: # 遍歷 final_boxes 中的每個視窗
    ax.add_patch(plt.Rectangle((i, j), Ni, Nj, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none')) # 在每個視窗的位置添加一個紅色的矩形框，表示篩選後的視窗
    


