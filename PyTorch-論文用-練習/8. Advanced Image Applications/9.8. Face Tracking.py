# 安裝套件： pip install face-recognition
'''
# 載入相關套件
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle, Circle  # 從 matplotlib.patches 匯入 Rectangle 和 Circle，用來在圖像上繪製矩形和圓形
import face_recognition  #進行人臉辨識和處理
import cv2  # 處理影像和影片的相關功能

# 載入圖檔
image_file = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\lin-manuel-miranda.png"  
image = plt.imread(image_file)  # imread() 讀取圖像檔案並將其儲存為numpy陣列格式的影像資料

plt.imshow(image)  
plt.axis('off')  
plt.show()  



# 載入影片檔
input_movie = cv2.VideoCapture(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\short_hamilton_clip.mp4")  
# 使用 cv2.VideoCapture() 讀取影片檔案，並將其儲存在變數 input_movie 中以供後續處理

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))  # 取得影片的總幀數，並將其轉換為整數後存入變數 length 中。
# 使用 cv2.CAP_PROP_FRAME_COUNT 來獲取影片總幀數

print(f'影片幀數：{length}')  

# 指定輸出檔名
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
# cv2.VideoWriter_fourcc() 設定影片的編碼格式，這裡使用 'XVID' 編碼，並將其儲存在變數 fourcc 中
# "*'XVID'"" 是解包操作符，將 'XVID' 字串解包成四個單獨的字元並傳入函數

# 每秒幀數(fps):29.97，影片解析度(Frame Size)：(640, 360)
output_movie = cv2.VideoWriter('./images_face/output.avi', fourcc, 29.97, (640, 360))  
# 使用 cv2.VideoWriter() 來創建影片寫入物件，指定輸出檔名、編碼格式、每秒幀數 (fps) 和影片解析度
# 這個物件將用來將處理後的影片幀寫入到新的影片檔案中

# 載入要辨識的圖像
image_file = 'lin-manuel-miranda.png'  
lmm_image = face_recognition.load_image_file("C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_face\\"+image_file)  
# 使用 face_recognition.load_image_file() 來載入指定的圖像檔案，並將其儲存在 lmm_image 變數中

# 取得圖像編碼
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]  
# 使用 face_recognition.face_encodings() 取得該圖像中所有偵測到的臉部特徵編碼
# 由於這張圖只包含一張臉，使用 [0] 來取出第一個臉部特徵編碼，並將其儲存在 lmm_face_encoding 中




# obama
image_file = 'obama.jpg'  
obama_image = face_recognition.load_image_file("C:\\Users\\user\\Desktop\\Python\\PyTorch\\Pytorch data\\images_face\\"+image_file)  
# face_recognition.load_image_file() 載入 obama 的圖像檔案，並儲存在 obama_image 變數中

# 取得圖像編碼
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]  
# face_recognition.face_encodings() 取得 obama 圖像中的臉部特徵編碼
# 圖中只有一張臉，因此直接取第一個編碼 [0]，並儲存在 obama_face_encoding 中

# 設定陣列
known_faces = [
    lmm_face_encoding,
    obama_face_encoding
]  
# 將兩個臉部特徵編碼(lmm 和 obama)儲存在列表 known_faces 中
# 這個列表將作為後續辨識過程中的已知臉部參考資料

# 目標名稱
face_names = ['lin-manuel-miranda', 'obama']  
# 設定臉部名稱列表，對應於已知臉部特徵編碼的名字
# 在辨識過程中，當匹配到某個臉部特徵時，會返回對應的名字

# 變數初始化
face_locations = []  # 用來儲存每幀影像中偵測到的臉部位置
face_encodings = []  # 用來儲存每幀影像中偵測到的臉部編碼
face_names = []  # 用來儲存每幀影像中辨識到的臉部名稱
frame_number = 0  # 初始化幀數計數器，用來追蹤當前處理到第幾幀影像


# 偵測臉部並寫入輸出檔
while True:  # 使用 while True 進行無限迴圈，直到手動中止或在迴圈內使用 break 來跳出
    # 讀取一幀影像
    ret, frame = input_movie.read()  
    # 使用 input_movie.read() 從影片中讀取一幀影像，返回兩個值：ret 表示是否成功讀取，frame 是讀取到的影像
    frame_number += 1  # 追蹤當前處理的幀數

    # 影片播放結束，即跳出迴圈
    if not ret:  
        break  # 如果 ret 為 False，表示影片已讀取完畢或發生錯誤，則跳出迴圈

    # 將 BGR 色系轉為 RGB 色系
    rgb_frame = frame[:, :, ::-1]  
    # OpenCV 讀取影像時預設為 BGR 色系，而 face_recognition 庫使用的是 RGB 色系
    # 這行使用切片操作將 BGR 轉換為 RGB，:::-1 表示反轉最後一個維度的順序

    # 找出臉部位置
    face_locations = face_recognition.face_locations(rgb_frame)  
    # face_recognition.face_locations() 來偵測圖像中的所有人臉，返回臉部位置的列表
    # 每個臉部位置是 (top, right, bottom, left) 的形式，分別對應臉部邊界的四個點

    # 編碼
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  
    # face_recognition.face_encodings() 根據已偵測的臉部位置來計算臉部編碼
    # 返回的 face_encodings 是一個包含每個臉部特徵編碼的列表

    # 比對臉部
    face_names = []  # 初始化空的名稱列表，來儲存每個偵測到的臉部對應的名稱
    for face_encoding in face_encodings:  # 對每一個臉部編碼進行迴圈處理
        # 比對臉部編碼是否與圖檔符合
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)  
        # face_recognition.compare_faces() 來將當前的臉部編碼與已知的臉部編碼進行比對
        # tolerance=0.50 設置了匹配的容忍度，數值越小匹配越嚴格

        # 找出符合臉部的名稱
        name = None  # 初始化變數 name 為 None，當匹配成功後會改變為對應的名稱
        for i in range(len(match)):  # 迴圈遍歷比對結果 match 列表
            if match[i] and 0 < i < len(face_names):  
                # 如果 match[i] 為 True，表示與某個已知臉部匹配成功
                # 並且 i 在 face_names 的索引範圍內
                name = face_names[i]  # 將匹配到的名稱儲存在變數 name 中
                break  # 跳出迴圈，因為已經找到匹配的名稱

        face_names.append(name)  # 將找到的名稱添加到 face_names 列表中

    # 輸出影片標記臉部位置及名稱
    for (top, right, bottom, left), name in zip(face_locations, face_names):  
        # 迴圈遍歷臉部位置和名稱，使用 zip 將兩者配對在一起
        if not name:  
            continue  # 如果 name 為 None，則跳過

        # 加框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  
        # cv2.rectangle() 在臉部周圍畫一個紅色矩形框
        # 矩形框的線條寬度為 2

        # 標記名稱
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)  
        # 在矩形框的下方畫一個紅色填充的矩形，用來顯示名稱的背景
        font = cv2.FONT_HERSHEY_DUPLEX  # 設定字型為 OpenCV 提供的字型：Hershey Duplex
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)  
        # cv2.putText() 在剛畫的紅色矩形框內輸出名稱
        # 名稱的顏色為白色 (255, 255, 255)，字型大小為 0.5，線條寬度為 1

    # 將每一幀影像存檔
    print("Writing frame {} / {}".format(frame_number, length))  
    # print出目前正在處理的幀數，追蹤進度
    output_movie.write(frame)  # output_movie.write() 將標註了臉部的影像幀寫入到輸出影片檔案中

# 關閉輸入檔    
input_movie.release()  
# input_movie.release() 釋放影片檔案資源，關閉影片檔案

# 關閉所有視窗
cv2.destroyAllWindows()  
# cv2.destroyAllWindows() 關閉所有由 OpenCV 開啟的視窗


'''



