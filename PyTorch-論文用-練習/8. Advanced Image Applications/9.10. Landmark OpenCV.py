'''
OpenCV針對臉部特徵的偵測，提供三種演算法

1. FacemarkLBF (Local Binary Features)
    FacemarkLBF 是基於局部二值特徵(LBF)和回歸樹的一種特徵點偵測算法。
    它通過學習如何從初始臉部形狀迭代地逼近真實的特徵點位置。
       
    局部二值特徵 (LBF)：
        這些是基於圖像的局部像素強度差異的簡單特徵，可以快速計算，適合實時應用。

    回歸樹：
        回歸樹是用於學習特徵點位置偏移的模型。
        每棵樹通過多次分裂來逼近目標值，最終匯集所有樹的結果來預測特徵點位置。

        1. 初始化形狀： 
            在圖像中找到臉部後，使用預定義的初始形狀 (如平均臉部形狀)作為初始猜測。

        2. 提取特徵：
            在臉部圖像的每個候選特徵點周圍，計算局部二值特徵。
            這些特徵是由像素間的灰度差異生成的二值模式。
        
        3. 回歸樹預測：
            使用預先訓練的回歸樹模型，基於提取的特徵預測每個特徵點的偏移量，並更新臉部形狀。

        4. 重複迭代：
            重複提取特徵並更新形狀，直到形狀收斂到最終的預測位置。
        

            
2. FacemarkAAM (Active Appearance Model)

    1. 初始化形狀：
    當給定一張新圖像時，AAM會首先使用一個初始形狀來作為起點。這通常是根據平均臉部形狀來設置的。
    
    2. 紋理提取與建模：
        根據初始化的形狀，AAM會在圖像中提取出對應的紋理(像素值)，並將其與模型中的平均紋理進行比較。

    3. 優化過程：
        AAM 的目標是最小化實際圖像中的紋理與模型生成的紋理之間的差異。這個過程通過調整形狀和紋理的參數來實現。
        通常使用梯度下降等優化算法來逐步調整形狀和紋理參數，使得模型生成的臉部形狀與紋理越來越接近真實圖像。

    4. 收斂到最終結果：
        當優化過程中的參數變化不再顯著(收斂)，AAM會輸出最終的特徵點位置。
        這些位置就是圖像中臉部各個特徵點的精確坐標。

        1. 形狀模型 (Shape Model)：
            形狀表示：形狀是由一組臉部特徵點 (如眼睛、鼻子、嘴巴等) 的位置構成的向量。
            每個特徵點都有 x 和 y 坐標。這些坐標組合起來，描述了臉部的形狀。

        2. 主成分分析 (PCA)：
            PCA 是一種統計方法，用來從多個臉部形狀數據中提取主要的形狀變化模式(降維)。
            這些模式能夠描述臉部在不同情況下 (如表情、角度) 如何變化。

        3. 紋理模型 (Texture Model)：
            1. 紋理表示：
                紋理指的是在標準形狀內部的像素值。
                AAM 會在每個特徵點之間形成的區域內提取紋理信息，並將這些信息轉換為一個紋理向量。
            2.PCA 紋理建模：
                與形狀建模類似，AAM 也會對紋理進行 PCA 分析，從而提取出主要的紋理變化模式。

        4. 聯合模型 (Combined Model)：
            聯合建模：AAM 結合形狀和紋理的變化，使用聯合的 PCA 分析來建立一個統一的模型，這樣可以同時考慮到形狀和紋理之間的關聯。

    【註】
    回歸樹 (Regression Tree)
        用於回歸問題，即用於預測連續值。例如預測股價

        1. 與決策樹類似，回歸樹的每個內部節點也是一個特徵的測試，但在回歸樹中，分裂後的節點仍然會對應連續的值而不是類別。
        2. 每次分裂都是基於最小化每個節點的均方誤差 (MSE)。這樣做的目的是使每個子集中的樣本的輸出值越相似越好。(使分裂後的兩個子集之間的差異最小) 
        3. 在每個葉節點處，回歸樹不再輸出類別標籤，而是輸出一個數值。這個數值通常是該葉節點內所有樣本的平均值。

    
            
3. FacemarkKamezi
    FacemarkKazemi 是基於逐級回歸(Cascaded Regression)的臉部特徵點偵測演算法。

    逐級回歸(Cascaded Regression)
        逐級回歸是一種迭代優化方法，主要思想是逐步逼近真實的特徵點位置。
        通過一系列的回歸步驟，每一步都基於上一階段的預測，應用一個新的回歸模型來進一步修正特徵點的位置。

    1. 初始形狀估計：
        在開始階段，FacemarkKazemi 使用一個預定義的初始形狀，通常是所有訓練數據集中臉部形狀的平均值。
        這個初始形狀會放置在已偵測到的臉部框架內，並作為初始特徵點位置的猜測。

    2. 特徵提取：
        在每個迭代步驟中，演算法會從當前估計的特徵點位置周圍提取局部圖像特徵。
        這些特徵可能包括灰度值、梯度信息等，目的是捕捉特徵點周圍的局部圖像特徵模式。

    3. 回歸模型應用：
        對於每個特徵點，使用訓練過的回歸樹來預測該特徵點從當前位置應該移動的偏移量。
        這些回歸樹是通過訓練數據學習到的，每棵樹根據輸入的圖像特徵計算出特徵點的位移。
        每一步的回歸結果都會修正當前的特徵點位置，這樣隨著每一步的迭代，特徵點位置越來越接近真實位置。

    4. 多步驟迭代：
        逐級回歸由多個回歸步驟組成，通常這些步驟很少但非常有效。
        每個步驟應用一組新的回歸樹來進一步優化特徵點位置，直至演算法達到預設的迭代次數或當特徵點位置收斂。

    5. 結果：
        最後一次迭代的結果即為最終估計的特徵點位置。這些點應該非常接近於真實的特徵點位置，即使初始估計值存在一定偏差。

'''

# 解除安裝套件： pip uninstall opencv-python opencv-contrib-python
# 安裝套件：    pip install opencv-contrib-python

# 載入相關套件
import cv2  
import numpy as np  
from matplotlib import pyplot as plt  

# 載入圖檔
image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_Object_Detection\lena.jpg" 
image = cv2.imread(image_file)  # cv2.imread() 讀取圖像文件，將圖像讀取為一個多維的 NumPy 數組，其中每個元素代表圖像的一個像素值。

# 顯示圖像
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
# OpenCV 是 BGR ，Matplotlib 是 RGB 
# cv2.cvtColor() 將 BGR 轉為 RGB

plt.imshow(image_RGB)  
plt.axis('off')  
plt.show()  



# 偵測臉部
cascade = cv2.CascadeClassifier(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\cascade_files\haarcascade_frontalface_alt2.xml")  
# cv2.CascadeClassifier() 加載 Haar Cascade 分類器檔案，預訓練的人臉檢測模型
# 模型基於 Haar 特徵，用來檢測臉部在圖像中的位置

faces = cascade.detectMultiScale(image , 1.5, 5)  
# detectMultiScale() 來偵測圖像中的臉部，返回一個包含臉部區域的矩形列表
# 第一個參數是輸入圖像，第二個參數是縮放比例(scaleFactor)，設定為 1.5
# 第三個參數是最少鄰居數(minNeighbors)，用來抑制多重檢測，設定為 5

print("faces", faces)  # 輸出偵測到的臉部位置，faces 是一個包含 (x, y, w, h) 的列表，代表每個臉部的矩形框

# 建立臉部特徵點偵測的物件
facemark = cv2.face.createFacemarkLBF()  
# cv2.face.createFacemarkLBF() 創建一個 LBF (Local Binary Features) 基於局部二元特徵的臉部標誌點偵測物件
# 這個物件將用來檢測臉部的特徵點，比如眼睛、鼻子、嘴巴等


# 訓練模型 lbfmodel.yaml 下載自：
# https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
facemark.loadModel(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\OpenCV\lbfmodel.yaml")  
# loadModel() 加載訓練好的 LBF 模型，該模型包含了特徵點檢測所需的數據，可以直接使用來檢測臉部的特徵點

# 偵測臉部特徵點
ok, landmarks1 = facemark.fit(image , faces)  
# fit() 函數對圖像中的臉部進行特徵點檢測
# 第一個參數是輸入圖像，第二個參數是臉部位置 (由 detectMultiScale 返回的矩形框列表)
# 這個函數返回兩個值：ok 是布林值，表示檢測是否成功、landmarks1 是特徵點的座標列表

print("landmarks LBF", ok, landmarks1)  # 輸出檢測結果，包含成功與否(ok)和特徵點座標列表(landmarks1)

# 繪製特徵點
for p in landmarks1[0][0]:  # 迴圈遍歷第一個臉部的特徵點座標列表
    cv2.circle(image, tuple(p.astype(int)), 5, (0, 255, 0), -1)  
    # cv2.circle() 在每個特徵點的位置畫一個綠色圓點
    # tuple(p.astype(int)) 將特徵點座標轉換為整數並轉換為元組，以符合 cv2.circle() 的參數要求
    # 圓點的半徑設定為 5，顏色為綠色 (0, 255, 0)，-1 表示填充整個圓


image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
# 再次將 BGR 轉換 RGB，以便 Matplotlib 顯示

plt.imshow(image_RGB)  
plt.axis('off')  
plt.show()  




# 建立臉部特徵點偵測的物件
facemark = cv2.face.createFacemarkAAM()  
# cv2.face.createFacemarkAAM() 創建一個 AAM (Active Appearance Model) 臉部特徵點偵測物件
# AAM 是一種基於外觀模型的臉部特徵點偵測技術，它結合了形狀和外觀資訊來進行精確的臉部特徵點檢測

# 訓練模型 aam.xml 下載自：
# https://github.com/berak/tt/blob/master/aam.xml
facemark.loadModel(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\OpenCV\aam.xml")  
# loadModel() 加載 AAM 模型，該模型包含了用於臉部特徵點檢測的預訓練數據

# 偵測臉部特徵點
ok, landmarks2 = facemark.fit(image , faces)  
# fit() 函數來對圖像中的臉部進行特徵點檢測
# 第一個參數是輸入圖像，第二個參數是臉部位置 (detectMultiScale 返回的矩形框列表)
# 這個函數返回兩個值：ok 是布林值，表示檢測是否成功、landmarks2 是特徵點的座標列表

print("Landmarks AAM", ok, landmarks2)  
# 輸出檢測結果，包含成功與否(ok)和特徵點座標列表(landmarks2)

# 繪製特徵點
for p in landmarks2[0][0]:  # 迴圈遍歷第一個臉部的特徵點座標列表
    cv2.circle(image, tuple(p.astype(int)), 5, (0, 255, 0), -1)  
    # cv2.circle() 在每個特徵點的位置畫一個綠色圓點
    # tuple(p.astype(int)) 將特徵點座標轉換為整數並轉換為元組，以符合 cv2.circle() 的參數要求
    # 圓點的半徑設定為 5，顏色為綠色 (0, 255, 0)，-1 表示填充整個圓


image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
# BGR 轉為 RGB

plt.imshow(image_RGB)  
plt.axis('off') 
plt.show()  





# 建立臉部特徵點偵測的物件
facemark = cv2.face.createFacemarkKazemi()  
# cv2.face.createFacemarkKazemi() 創建一個 Kazemi 臉部特徵點偵測物件
# Kazemi 演算法是一種快速且高效的臉部特徵點檢測技術，通常比其他方法具有更高的效能

# 訓練模型 face_landmark_model.dat 下載自：
# https://github.com/opencv/opencv_3rdparty/tree/contrib_face_alignment_20170818
facemark.loadModel("./OpenCV/face_landmark_model.dat")  
# 使用 loadModel() 加載 Kazemi 模型，該模型包含了用於臉部特徵點檢測的預訓練數據
# 模型需要事先下載並放置於合適的位置，然後可以通過路徑加載來使用

# 偵測臉部特徵點
ok, landmarks2 = facemark.fit(image , faces)  
# 使用 fit() 函數來對圖像中的臉部進行特徵點檢測
# 第一個參數是輸入圖像，第二個參數是臉部位置（由 detectMultiScale 返回的矩形框列表）
# 這個函數返回兩個值：ok 是布林值，表示檢測是否成功；landmarks2 是特徵點的座標列表

print("Landmarks Kazemi", ok, landmarks2)  
# 輸出檢測結果，包含成功與否的標誌（ok）和特徵點座標列表（landmarks2）

# 繪製特徵點
for p in landmarks2[0][0]:  # 迴圈遍歷第一個臉部的特徵點座標列表
    cv2.circle(image, tuple(p.astype(int)), 5, (0, 255, 0), -1)  
    # 使用 cv2.circle() 在每個特徵點的位置畫一個綠色圓點
    # tuple(p.astype(int)) 將特徵點座標轉換為整數並轉換為元組，以符合 cv2.circle() 的參數要求
    # 圓點的半徑設定為 5，顏色為綠色 (0, 255, 0)，-1 表示填充整個圓

# 顯示圖像
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

plt.imshow(image_RGB)  
plt.axis('off')  
plt.show()  



