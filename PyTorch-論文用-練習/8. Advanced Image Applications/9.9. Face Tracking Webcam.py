'''
1. Webcam： 
    捕捉連續的影像幀。

2. 臉部檢測：
    每幀需要檢測是否有臉部存在。如果有，需要確定每個臉部的位置。
    通過 CNN 檢測不同角度和光照下的人臉。

3. 臉部特徵提取：
    特徵通常以128維的向量表示。

4. 臉部識別：
    將提取到的臉部特徵與預先存儲在系統中的已知臉部特徵進行比較，以識別這張臉屬於誰。
    如果兩個特徵向量之間的距離小於某個閾值，那麼我們就認為這是同一個人。
'''
'''
# 安裝套件： pip install face-recognition

# 載入相關套件
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle, Circle  # 從 matplotlib.patches 匯入 Rectangle 和 Circle，用來在圖像上繪製矩形和圓形
import face_recognition  

# 載入圖檔
image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\classmates.jpg" 
image = plt.imread(image_file)  # plt.imread() 讀取圖像檔案並將其儲存為陣列格式的影像資料

plt.imshow(image) 
plt.axis('off')  
plt.show() 


# 偵測臉部特徵點並顯示
from PIL import Image, ImageDraw  # 從 PIL 匯入 Image 和 ImageDraw，Image 用來處理圖像，ImageDraw 用來在圖像上進行繪圖操作

# 載入圖檔
image = face_recognition.load_image_file(image_file)  
# face_recognition.load_image_file() 來載入圖像檔案，並將其儲存為 numpy 陣列格式

# 轉為 Pillow 圖像格式
pil_image = Image.fromarray(image)  
# Image.fromarray() 函數將 numpy 陣列格式的圖像轉換為 PIL Image 格式，這樣可以使用 PIL 的繪圖功能來操作圖像

# 取得圖像繪圖物件
d = ImageDraw.Draw(pil_image)  
# ImageDraw.Draw() 創建一個繪圖物件 d，該物件可以用來在 pil_image 上進行繪圖操作

# 偵測臉部特徵點
face_landmarks_list = face_recognition.face_landmarks(image)  
# face_recognition.face_landmarks() 函數來偵測圖像中所有臉部的特徵點
# 返回的 face_landmarks_list 是一個列表，列表中的每個元素是一個字典，包含該臉部的各個特徵點位置

for face_landmarks in face_landmarks_list:  # 迴圈遍歷每個偵測到的臉部特徵點字典
    # 顯示五官特徵點
    for facial_feature in face_landmarks.keys():  # 迴圈遍歷每個臉部特徵 (例如嘴巴、眼睛、鼻子)
        print(f"{facial_feature} 特徵點: {face_landmarks[facial_feature]}\n")  


    # 繪製特徵點
    for facial_feature in face_landmarks.keys():  # 再次遍歷臉部特徵
        d.line(face_landmarks[facial_feature], width=5, fill='green')  
        # d.line() 在特徵點之間畫線來連接它們，形成可視化的特徵線條，width=5 設定線條寬度為 5，fill='green' 將線條顏色設定為綠色
    
plt.imshow(pil_image)  
plt.axis('off')  
plt.show()  





import dlib  # 用來進行人臉偵測和特徵點定位
import cv2  # 處理影像和影片的相關功能
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle, Circle  # 用來在圖像上繪製矩形和圓形
from imutils import face_utils  # 從 imutils 庫匯入 face_utils 模組，用來將 dlib 偵測的臉部特徵點轉換為 NumPy 陣列格式

# 載入圖檔
image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\classmates.jpg" 
image = plt.imread(image_file)  # plt.imread() 讀取圖像檔案並將其儲存為陣列格式的影像資料

plt.imshow(image)  
plt.axis('off')  
plt.show()  

# 載入 dlib 以 HOG 基礎的臉部偵測模型
model_file = "shape_predictor_68_face_landmarks.dat"  
# 指定形狀預測器的模型檔案，這是基於 HOG (Histogram of Oriented Gradients) 的臉部偵測和特徵點預測模型

detector = dlib.get_frontal_face_detector()  
# dlib.get_frontal_face_detector() 來獲取臉部偵測器，這是一個基於 HOG 的臉部偵測方法

predictor = dlib.shape_predictor(model_file)  
# dlib.shape_predictor() 加載形狀預測器模型，用來偵測臉部的 68 個特徵點

# 偵測圖像的臉部
rects = detector(image)  
# detector(image) 來偵測圖像中的所有臉部，返回的 rects 是一個 dlib 矩形物件列表，表示每個偵測到的臉部的位置

print(f'偵測到{len(rects)}張臉部.')  


# 偵測每張臉的特徵點
for (i, rect) in enumerate(rects):  
    # 迴圈遍歷每個偵測到的臉部，enumerate() 提供索引 i 和對應的 rect 物件
    # 偵測特徵點
    shape = predictor(image, rect)  
    # predictor() 函數對每個臉部位置 rect 進行特徵點預測，返回 shape 是一個包含 68 個特徵點位置的 dlib 物件
    
    # 轉為 NumPy 陣列
    shape = face_utils.shape_to_np(shape)  
    # face_utils.shape_to_np() 將 dlib 的 shape 物件轉換為 NumPy 陣列，便於後續的圖像處理
    
    # 標示特徵點
    for (x, y) in shape:  # 迴圈遍歷每個特徵點的 x, y 座標
        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)  
        # cv2.circle() 在每個特徵點位置畫一個半徑為 10 的綠色圓點
        # (0, 255, 0) 代表 BGR 色系中的綠色，-1 表示圓的填充模式

plt.imshow(image)  
plt.axis('off')  
plt.show()  

# 讀取視訊檔
cap = cv2.VideoCapture(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\hamilton_clip.mp4')  
# cv2.VideoCapture() 讀取影片檔案，並將其儲存在變數 cap 中以供後續處理

while True:  
    # 讀取一幀影像
    _, image = cap.read()  
    # cap.read() 從影片中讀取一幀影像
    # _ 忽略第一個返回值(ret，是否成功)，image 是讀取到的影像
    
    # 偵測圖像的臉部
    rects = detector(image)  
    # detector(image) 來偵測影像中的所有臉部，返回 rects 是一個包含臉部矩形位置的列表
    
    for (i, rect) in enumerate(rects):  # 迴圈遍歷每個偵測到的臉部
        # 偵測特徵點
        shape = predictor(image, rect)  
        # predictor() 對每個臉部位置 rect 進行特徵點預測
        shape = face_utils.shape_to_np(shape)  
        # face_utils.shape_to_np() 將特徵點轉換為 NumPy 陣列
        
        # 標示特徵點
        for (x, y) in shape:  # 迴圈遍歷每個特徵點的 x, y 座標
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  
            # cv2.circle() 在每個特徵點位置畫一個半徑為 2 的綠色圓點

    # 顯示影像
    cv2.imshow("Output", image)  
    # cv2.imshow() 顯示影像視窗，標題為 "Output"，顯示的是當前處理的影像幀

    k = cv2.waitKey(5) & 0xFF  # 按 Esc 跳離迴圈
    # 使用 cv2.waitKey(5) 等待 5 毫秒並捕捉鍵盤輸入，返回值與 0xFF 進行按位與操作，以保證結果在各平台上保持一致
    # 如果按下 Esc 鍵 (對應的 ASCII 值為 27)，則退出迴圈
    if k == 27:  # 27 是 Esc 鍵的 ASCII 值
        break  # 跳出 while 迴圈

# 關閉輸入檔    
cap.release()  # cap.release() 釋放影片檔案資源，關閉影片檔案

# 關閉所有視窗
cv2.destroyAllWindows()  
# cv2.destroyAllWindows() 關閉所有由 OpenCV 開啟的視窗


'''


