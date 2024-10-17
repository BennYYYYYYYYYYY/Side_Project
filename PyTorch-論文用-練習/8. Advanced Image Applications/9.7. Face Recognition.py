'''
1. 臉部追蹤 (Face Tracking)

    臉部追蹤指的是在影像序列中持續地檢測並跟踪人臉的位置
    
    1. 臉部檢測 (Face Detection)
        識別出影像中所有人臉的位置。通常是通過CNN

    2. 臉部識別 (Face Recognition)
        在識別出人臉後，可以進一步將每個人臉與已知的人臉數據庫進行比對，以識別出具體的人物。

    3. 臉部追踪 (Face Tracking)
        在連續的影像中，根據臉部的位置和特徵來保持對特定人臉的追踪。

        

2. Face Recognition： 

    它是一個Python庫，用於進行臉部識別。是用dlib為基礎的套件，它使用C++開發，所以要先安裝dlib套件。
    它基於深度學習模型 (如ResNet) 和 dlib 庫，提供了簡單易用的 API 來進行臉部檢測和識別。
    
    主要功能：
    1. 臉部檢測：自動檢測影像或視訊中的所有人臉。
    2. 臉部特徵提取：提取臉部的128維特徵向量，這是一個臉部的唯一表示。
    3. 臉部識別：通過比較特徵向量來識別和匹配臉部。
        
    原理：
    1. 人臉檢測模型：
        face-recognition 使用的檢測模型基於 dlib 中的 Histogram of Oriented Gradients (HOG) 和 CNN。

    2. 臉部編碼：
        指將臉部圖像轉換為固定長度的向量，這個向量稱為特徵向量。特徵向量保留了臉部的關鍵信息。

    3. 臉部匹配：
        臉部匹配是通過比較兩個臉部的特徵向量之間的歐氏距離來實現的。如果距離足夠小，則可以認為兩張臉屬於同一個人。

'''
'''
# 安裝套件： pip install face-recognition
# 載入相關套件
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle, Circle  # matplotlib.patches 導入 Rectangle和Circle類，用於在圖像上添加矩形和圓形標記。
import face_recognition  # 用於人臉識別。


image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\classmates.jpg"  
image = plt.imread(image_file)  # imread 載入圖像並存儲為一個NumPy陣列，每個像素的顏色值會被存儲起來。

plt.imshow(image)  
plt.axis('off')  
plt.show() 


# 偵測臉部
faces = face_recognition.face_locations(image)  # 使用face_recognition.face_locations()函數來檢測圖像中的人臉位置。
# 返回一個包含每個檢測到的人臉位置的列表，每個位置由一個tuple表示，格式為(top, right, bottom, left)。

# 臉部加框
ax = plt.gca()  # 使用plt.gca()函數獲取當前的軸對象(ax)，這樣可以在圖像上繪製矩形框。
for result in faces:  # 遍歷每一個檢測到的人臉位置。
    # 取得框的座標
    y1, x1, y2, x2 = result  # 解壓縮每個人臉位置的tuple，分別獲取上(y1)、右(x2)、下(y2)、左(x1)邊界的坐標。
    width, height = x2 - x1, y2 - y1  # 計算矩形框的寬度和高度，寬度為右邊界減去左邊界，高度為下邊界減去上邊界。
    # 加紅色框
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')  # 創建一個矩形對象，左上角為(x1, y1)，寬度和高度分別為`width`和`height`。
    # fill=False表示矩形不填充顏色，color='red'設置矩形邊框為紅色。

    ax.add_patch(rect)  # 使用ax.add_patch()方法將矩形添加到當前的軸對象上。

plt.imshow(image)  
plt.axis('off') 
plt.show() 



# 偵測臉部特徵點並顯示
from PIL import Image, ImageDraw  # 從 PIL 導入， Image用於處理圖像，ImageDraw用於在圖像上繪製。

image = face_recognition.load_image_file(image_file)  # 使用face_recognition的load_image_file()來載入圖像檔案，返回一個包含圖像數據的numpy數組。

pil_image = Image.fromarray(image)  # 使用PIL的Image.fromarray()將numpy數組轉換為Pillow圖像對象，這樣可以利用Pillow進行進一步的圖像處理。

# 取得圖像繪圖物件
d = ImageDraw.Draw(pil_image)  # 使用ImageDraw.Draw()創建一個繪圖對象"d"，可以用來在"pil_image"這張圖像上繪製形狀或線條。

# 偵測臉部特徵點
face_landmarks_list = face_recognition.face_landmarks(image)  # 使用face_recognition.face_landmarks()檢測圖像中的臉部特徵點，返回一個包含每個臉部特徵點字典的列表。
# 每個字典中包含五官特徵點(如眉毛、眼睛、鼻子、嘴唇等)的名稱和座標。


for face_landmarks in face_landmarks_list:  # 遍歷每一個臉部特徵點的字典。
    # 顯示五官特徵點
    for facial_feature in face_landmarks.keys():  # 遍歷臉部特徵點字典中的每個五官特徵名稱。
        print(f"{facial_feature} 特徵點: {face_landmarks[facial_feature]}\n")  # 輸出五官特徵的名稱及其對應的座標。

    # 繪製特徵點
    for facial_feature in face_landmarks.keys():  # 再次遍歷臉部特徵點字典中的每個五官特徵名稱。
        d.line(face_landmarks[facial_feature], width=5, fill='green')  # 使用d.line()函數來繪製線條，將每個五官特徵點的座標連接起來。
        # face_landmarks[facial_feature] 是該五官特徵的所有座標點
        # width=5 設置線條寬度為5
        # fill='green' 設置線條顏色為綠色

plt.imshow(pil_image)  # 顯示繪製完五官特徵點的圖像
plt.axis('off')  
plt.show()  


'''
