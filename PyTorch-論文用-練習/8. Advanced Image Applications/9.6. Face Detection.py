'''
1. 臉部偵測 (Face Detection)
    臉部偵測的主要目標是從背景中分離出臉部區域，這一步為後續的臉部辨識或其他分析打下基礎。


2. 臉部偵測技術分類
    1. 基於特徵的演算法: Haar Cascades
    2. 基於深度學習的方法: MTCNN


3. Haar Cascades 演算法
    該算法通過訓練一個 級聯分類器(Cascade Classifier) 來快速、有效地檢測圖像中的對象 (如人臉)。
    由於其較高的效率，至今仍被廣泛應用於一些資源受限的環境中(例如手機和嵌入式系統)。

    
4. Haar-like 特徵
    Haar-like 特徵基於簡單的矩形區域對比來描述圖像的局部特徵，這些特徵可以捕捉到圖像中的邊緣、線條、和其他幾何形狀。
    這些矩形區域分為兩個部分，分別計算區域內的像素值總和，然後將這兩個總和相減來得到一個特徵值。這個特徵值表示了該區域內的亮度變化。

        1. 邊緣特徵
            例如一個白色矩形和一個黑色矩形相鄰，這種結構可以捕捉到圖像中的邊緣。
            這些邊緣可以是水平、垂直或斜向的。
        
        2. 線條特徵
            例如兩個黑色矩形夾著一個白色矩形，可以檢測水平或垂直的線條。

        3. 四角特徵
            這種特徵由四個矩形組成，通常用於檢測物體的角點。

    這些特徵非常簡單，但數量龐大，因此在一幅圖像上應用所有可能的特徵會產生巨大的計算負擔。
    為了解決這個問題，Haar Cascades 採用了級聯分類器 (Cascade Classifier) 和積分圖像 (Integral Image)的技術。


5. 積分圖像 (Integral Image)
    積分圖像是 Haar Cascades 中用來加速特徵計算的關鍵技術。
    它的主要目的是使得任意矩形區域內的像素和可以快速計算，這樣在大量特徵計算中就可以大幅度提高效率。

    積分圖像是一個與原始圖像大小相同的矩陣。對於原始圖像中的每一個像素位置 (x,y)，
    積分圖像中的值II(x,y)是該像素點左上角(包括該點)所有像素值的總和。公式表示：
        II(x,y) = ∑ ∑ I(i,j)
            I(i,j) = 原始圖像在位置(i,j)的像素值。

    積分圖像的計算是從圖像的左上角開始，逐行逐列計算每個點的積分值。


6. 級聯分類器 (Cascade Classifier)
    級聯分類器是 Haar Cascades 演算法的另一個核心組件，它通過將多個弱分類器串聯在一起，形成一個強分類器來實現。
    級聯分類器的設計理念是通過逐層篩選來高效地過濾出潛在的目標區域 (人臉)，從而避免對所有圖像區域進行繁瑣的計算。

        1. 弱分類器 (Weak Classifier)
            在級聯分類器中，每一層都是一個弱分類器。
            這些弱分類器通常基於簡單的決策樹，它們的作用是針對一個特定的 Haar-like 特徵進行判斷。
            雖然單個弱分類器的準確性不高，但它們能夠快速地排除明顯不包含目標 (人臉)的區域。

    級聯結構的分層設計

        1. 初級篩選(第一層)
            這一層的分類器簡單且高效，用來篩選出絕大多數不可能包含人臉的區域。它使用的特徵數量很少，因此計算量低。

        2. 中間層級
            隨著層數的增加，分類器會變得更加複雜，每一層使用的特徵數量會增加，這樣可以更精確地識別可能包含人臉的區域。

        3. 高級篩選(最後幾層)
            這些層次中的分類器非常細緻，它們會針對那些高度懷疑為人臉的區域進行全面的檢查。
            這些層次使用了大量的特徵，從而保證最終的檢測結果準確無誤。

    級聯分類器的工作流程

        1. 初步判斷
            每個圖像區域首先經過第一層的分類器。如果該區域通過了第一層，則進入下一層檢查；否則被立即拋棄。

        2. 逐層判斷
            如果一個圖像區域通過了當前層，則進入下一層。隨著層數的增加，分類器會越來越嚴格，使用的特徵也會越來越多。
        
        3. 最終判定
            只有當一個區域通過了所有層次的檢查時，該區域才會被標記為包含人臉。
'''

# 匯入cv2模組
import cv2  
from cv2 import CascadeClassifier  # CascadeClassifier 用於對象檢測，通常用於臉部檢測。
from cv2 import rectangle  # rectangle 用來在圖像上繪製矩形。
import matplotlib.pyplot as plt  
from cv2 import imread  # imread 用來從檔案中讀取圖像並將其載入到程式中。

# 載入臉部級聯分類器(face cascade file)
face_cascade = r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\cascade_files\haarcascade_frontalface_alt.xml'  # 預訓練的XML文件，包含了用於臉部檢測的Haar級聯分類器模型的參數。
classifier = cv2.CascadeClassifier(face_cascade)  # 用CascadeClassifier類別來載入該級聯分類器，之後可以用它來檢測圖像中的臉部。

image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\teammates.jpg"  
image = imread(image_file)  # imread 載入圖像並存儲為一個NumPy陣列，每個像素的顏色值會被存儲起來。

# OpenCV 預設為 BGR 色系，轉為 RGB 色系
im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 讀取圖像的預設顏色順序是BGR，使用cv2.cvtColor函數將其轉換為RGB，因為Matplotlib使用的是RGB。

# 顯示圖像
plt.imshow(im_rgb)  # 顯示圖像 (轉換為RGB後的圖像)
plt.axis('off')  # 關閉軸顯示
plt.show()  

# 偵測臉部
bboxes = classifier.detectMultiScale(image)  # 使用級聯分類器的detectMultiScale方法來偵測圖像中的臉部。
# 會返回一個包含偵測到的臉部位置及大小的矩形框(bounding boxes)的列表。

# 臉部加框
for box in bboxes:  # 遍歷每一個偵測到的臉部框，為每一個臉部畫框。
    # 取得框的座標及寬高
    x, y, width, height = box  # 解析出每個框的左上角座標(x, y)以及框的寬度(width)和高度(height)。
    x2, y2 = x + width, y + height  # 計算框的右下角座標(x2, y2)。
    # 加白色框
    rectangle(im_rgb, (x, y), (x2, y2), (255,255,255), 2)  # rectangle 在圖像上畫出一個白色(255,255,255)的矩形框，框線寬度為2像素。

# 顯示圖像
plt.imshow(im_rgb)  
plt.axis('off')  
plt.show() 




# 載入圖檔
image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\classmates.jpg"  
image = imread(image_file)  # imread 載入新圖像。

im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR格式的圖像轉換為RGB。

plt.imshow(im_rgb)  
plt.axis('off')  
plt.show()  

# 偵測臉部
bboxes = classifier.detectMultiScale(image)  # 使用載入的臉部級聯分類器來偵測圖像中的臉部。
# detectMultiScale方法會返回所有偵測到的臉部的位置和大小(用矩形框表示)。

# 臉部加框
for box in bboxes:  # 遍歷每一個偵測到的臉部框，對每一個臉部進行加框處理。
    # 取得框的座標及寬高
    x, y, width, height = box  # 解析出每個框的左上角座標(x, y)以及框的寬度(width)和高度(height)。
    x2, y2 = x + width, y + height  # 計算框的右下角座標(x2, y2)。
    # 加紅色框
    rectangle(im_rgb, (x, y), (x2, y2), (255, 0, 0), 5)  # rectangle 在圖像上畫出一個紅色(255, 0, 0)的矩形框，框線寬度為5像素。

plt.imshow(im_rgb)  # 顯示加了紅色框的圖像
plt.axis('off')  
plt.show() 




# 載入眼睛級聯分類器(eye cascade file)
eye_cascade = r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\cascade_files\haarcascade_eye_tree_eyeglasses.xml'  # 用於偵測眼睛的級聯分類器的XML文件，適合於戴眼鏡的情況。
eye_classifier = cv2.CascadeClassifier(eye_cascade)  # 使用CascadeClassifier類別載入該級聯分類器，以後可以用來檢測圖像中的眼睛。

# 載入微笑級聯分類器(smile cascade file)
smile_cascade = r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\cascade_files\haarcascade_smile.xml'  # 用於偵測微笑的級聯分類器的XML文件。
smile_classifier = cv2.CascadeClassifier(smile_cascade)  # CascadeClassifier 載入級聯分類器，可以用來檢測圖像中的微笑。

image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\classmates.jpg"  
image = imread(image_file)  # imread 載入圖像

im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將BGR的圖像轉換為RGB

plt.imshow(im_rgb)  
plt.axis('off')  
plt.show()  



im_rgb_clone = im_rgb.copy()  # 創建一個圖像的副本。這樣可以在不改變原始圖像的情況下進行後續的操作。

# 偵測臉部
bboxes = classifier.detectMultiScale(image)  # 使用臉部級聯分類器再次檢測新圖像中的臉部。
# 臉部加框
for box in bboxes:  # 遍歷每一個偵測到的臉部框，對每一個臉部進行加框處理。
    x, y, width, height = box  # 取得框的座標及寬高
    x2, y2 = x + width, y + height  # 計算框的右下角座標(x2, y2)。
    # 加紅色框
    rectangle(im_rgb_clone, (x, y), (x2, y2), (255, 0, 0), 5)  # rectangle 在圖像copy上畫出紅色矩形框。


# 偵測微笑
# scaleFactor=2.5：掃描時每次縮減掃描視窗的尺寸比例。
# minNeighbors=20：每一個被選中的視窗至少要有鄰近且合格的視窗數
bboxes = smile_classifier.detectMultiScale(image, 2.5, 20)  # 使用微笑級聯分類器來偵測圖像中的微笑。scaleFactor參數控制掃描視窗縮減比例，minNeighbors參數確保偵測結果的準確性。
# 微笑加框
for box in bboxes:  # 遍歷每一個偵測到的微笑框，對每一個微笑進行加框處理。
    x, y, width, height = box  # 取得框的座標及寬高
    x2, y2 = x + width, y + height  # 計算框的右下角座標(x2, y2)。
    # 加紅色框
    rectangle(im_rgb_clone, (x, y), (x2, y2), (255, 0, 0), 5)  # 使用cv2的rectangle函數在圖像副本上畫出紅色矩形框來標示微笑。
#     break  # 如果取消註釋，這行代碼會在檢測到第一個微笑後跳出循環，這樣只會標示第一個偵測到的微笑。

# 顯示圖像
plt.imshow(im_rgb_clone)  # 加了臉部和微笑框的圖像copy。
plt.axis('off')  
plt.show()  



image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\classmates.jpg"  
image = imread(image_file)  
im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

plt.imshow(im_rgb)  
plt.axis('off')  
plt.show()  

im_rgb_clone = im_rgb.copy()  

# 偵測臉部
bboxes = classifier.detectMultiScale(image)  # 使用級聯分類器檢測圖像中的臉部
# detectMultiScale返回一個矩形框的列表，這些框表示偵測到的臉部。

# 臉部加框
for box in bboxes:  # 遍歷每一個偵測到的臉部框，對每一個臉部進行加框處理。
    # 取得框的座標及寬高
    x, y, width, height = box  # 解析出每個框的左上角座標(x, y)以及框的寬度(width)和高度(height)。
    x2, y2 = x + width, y + height  # 計算框的右下角座標(x2, y2)。
    # 加白色框
    rectangle(im_rgb_clone, (x, y), (x2, y2), (255, 0, 0), 5)  # rectangle 在圖像副本上畫出紅色(255, 0, 0)的矩形框，框線寬度為5像素。

    # 偵測眼睛
    face_box = image[y:y2, x:x2]  # 將偵測到的臉部框從原始圖像中截取出來，作為一個新的子圖像，用於進一步的眼睛和微笑檢測。
    bboxes_eye = eye_classifier.detectMultiScale(face_box, 1.1, 5)  # 使用眼睛級聯分類器來檢測臉部框內的眼睛。scaleFactor設為1.1，minNeighbors設為5。
    # 加框
    for box_eye in bboxes_eye:  # 遍歷偵測到的眼睛框，對每一個眼睛進行加框處理。
        # 取得框的座標及寬高
        x, y, width, height = box_eye  # 解析出每個眼睛框的左上角座標(x, y)以及框的寬度(width)和高度(height)。
        x2, y2 = x + width, y + height  # 計算框的右下角座標(x2, y2)。
        # 加紅色框
        rectangle(im_rgb_clone, (x+box[0], y+box[1]), (x2+box[0], y2+box[1]), (255, 0, 0), 5)  # 在臉部框內的位置上加框，需要將眼睛框的坐標偏移到整體圖像的座標系中。

    # 偵測微笑
    # scaleFactor=2.5：掃描時每次縮減掃描視窗的尺寸比例。
    # minNeighbors=20：每一個被選中的視窗至少要有鄰近且合格的視窗數
    bboxes_smile = smile_classifier.detectMultiScale(face_box, 2.5, 20, 0)  # 使用微笑級聯分類器來檢測臉部框內的微笑。scaleFactor設為2.5，minNeighbors設為20。參數0表示使用預設標誌。
    # 加框
    for box_smile in bboxes_smile:  # 遍歷偵測到的微笑框，對每一個微笑進行加框處理。
        # 取得框的座標及寬高
        x, y, width, height = box_smile  # 解析出每個微笑框的左上角座標(x, y)以及框的寬度(width)和高度(height)。
        x2, y2 = x + width, y + height  # 計算框的右下角座標(x2, y2)。
        # 加紅色框
        rectangle(im_rgb_clone, (x+box[0], y+box[1]), (x2+box[0], y2+box[1]), (255, 0, 0), 5)  # 同樣地，將微笑框的坐標偏移到整體圖像的座標系中，並在該位置加框。

plt.imshow(im_rgb_clone)  
plt.axis('off')  
plt.show()  


