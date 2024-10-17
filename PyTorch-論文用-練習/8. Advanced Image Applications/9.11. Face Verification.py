'''
Face verification
    Face verification用於確認兩張面部圖像是否屬於同一個人。
    
    1. 面部特徵提取(Feature Extraction)
        將人臉圖像轉換成一個可以用來進行比較的數值表示
        CNN 通過多層卷積過程學習到人臉的不同層次的特徵，並最終輸出一個固定大小的向量，即特徵向量(Feature Vector)

    2. 特徵向量比較
        從不同的面部圖像中提取出特徵向量後，接下來要比較這些向量，確定它們是否代表同一個人。

            1. 向量的相似性
                在高維空間中，兩個向量的相似性通常使用【距離度量】來表示。
                最常用的距離度量是歐幾里得距離(Euclidean Distance)：兩個向量之間的直線距離
                但對於人臉識別來說，餘弦相似度 (Cosine Similarity) 更常用，因為它專注於向量之間的夾角而不是它們的大小。

                Cosine Similarity = AB內積/A*B長度

                當餘弦相似度接近1時，表示兩個向量之間的夾角很小，兩個向量非常相似，這意味著兩張臉可能是同一個人。
                當餘弦相似度接近0時，則表示兩張臉不相似。

            2. 法向量(Normal Vector)
                在高維空間中（如三維以上），「平面」稱為超平面(Hyperplane)。它是能夠將空間分割成兩部分的一個子空間。
                例如，在二維空間中，直線是分割平面的「超平面」；在三維空間中，平面是分割空間的「超平面」。

                法向量是指垂直於某個超平面的向量。
                在使用支持向量機 (SVM) 進行分類時，法向量可以用來表示決策邊界的方向。

                SVM 尋找的是能夠最大化不同類別(如「同個人」和「不同人」)之間的間隔的超平面。
                這個超平面的法向量定義了決策邊界的方向，並且所有在法向量同一側的向量將被分類為同一類。

'''
'''
# 載入相關套件
import face_recognition  
import numpy as np 
from matplotlib import pyplot as plt  


known_image_1 = face_recognition.load_image_file(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\jared_1.jpg") # face_recognition.load_image_file() 讀取圖像檔案並將其儲存為陣列格式的影像資料
known_image_2 = face_recognition.load_image_file(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\jared_2.jpg")  
known_image_3 = face_recognition.load_image_file(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\jared_3.jpg")  
known_image_4 = face_recognition.load_image_file(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\obama.jpg")  

# 標記圖檔名稱
names = ["jared_1.jpg", "jared_2.jpg", "jared_3.jpg", "obama.jpg"]  

# 顯示圖像
unknown_image = face_recognition.load_image_file(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\jared_4.jpg") # 讀取要辨識的圖像"jared_4.jpg"，並儲存到 unknown_image 變數中

plt.imshow(unknown_image)  
plt.axis('off') 
plt.show() 


# 圖像編碼
known_image_1_encoding = face_recognition.face_encodings(known_image_1)[0]  
# face_recognition.face_encodings() 取得 jared_1.jpg 圖像中所有偵測到的臉部特徵編碼
# 使用 [0] 來取出第一個臉部特徵編碼，並將其儲存在 known_image_1_encoding 中

known_image_2_encoding = face_recognition.face_encodings(known_image_2)[0]  
known_image_3_encoding = face_recognition.face_encodings(known_image_3)[0]  
known_image_4_encoding = face_recognition.face_encodings(known_image_4)[0]  


known_encodings = [known_image_1_encoding, known_image_2_encoding, 
                   known_image_3_encoding, known_image_4_encoding]  
# 將所有已知圖像的臉部特徵編碼儲存到列表 known_encodings 中，這個列表將用作後續人臉辨識中的比較參考

unknown_encoding = face_recognition.face_encodings(unknown_image)[0]  
# 取得 jared_4.jpg 中的臉部特徵編碼，並儲存在 unknown_encoding 變數中


# 比對
results = face_recognition.compare_faces(known_encodings, unknown_encoding)  
# face_recognition.compare_faces() 來將未知圖像的臉部編碼與已知的臉部編碼進行比對
# 這裡傳入的參數為已知的編碼列表 (known_encodings) 和未知的臉部編碼 (unknown_encoding)
# 函數會返回一個布林值列表，每個值代表未知臉部是否與相應的已知臉部匹配

print(results)  
# print 出比對結果，這是一個包含布林值的列表，指示未知圖像是否與任何已知圖像匹配



# 載入相關套件
import dlib  # 用於人臉偵測和人臉特徵提取
import cv2  # 用於影像處理
import numpy as np  
from matplotlib import pyplot as plt 

# 載入模型
pose_predictor_5_point = dlib.shape_predictor(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\OpenCV\shape_predictor_5_face_landmarks.dat")  
# 載入 dlib 的 5 點臉部特徵模型，該模型用於提取人臉的關鍵特徵點 (如眼睛、鼻子等)

face_encoder = dlib.face_recognition_model_v1(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\OpenCV\dlib_face_recognition_resnet_model_v1.dat")  
# 載入 dlib 的 ResNet 人臉特徵編碼模型，該模型用於生成臉部特徵向量，這些向量代表人臉的特徵

detector = dlib.get_frontal_face_detector()  
# 載入 dlib 的前置臉部偵測器，用於偵測影像中的人臉位置

# 找出哪一張臉最相似
def compare_faces_ordered(encodings, face_names, encoding_to_check):
    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))  
    # 計算未知臉部編碼與每個已知臉部編碼之間的歐幾里得距離(L2距離)
    # np.linalg.norm 用來計算向量之間的距離，axis=1 表示對每個向量逐行計算

    return zip(*sorted(zip(distances, face_names)))  
    # 將距離和對應的名稱進行排序，依照相似度(距離由小到大)排序
    # sorted() 函數根據距離排序後，使用 zip() 將距離和名稱分開


# 利用線性代數的法向量比較兩張臉的特徵點
def compare_faces(encodings, encoding_to_check):
    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))  
    # 計算未知臉部編碼與所有已知臉部編碼的距離，並返回一個包含距離的列表
    # 這些距離表示未知臉部與每個已知臉部的相似程度，距離越小越相似


# 圖像編碼
def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    # 偵測臉部
    face_locations = detector(face_image, number_of_times_to_upsample)  
    # 使用前面載入的 detector 偵測臉部位置，返回一個包含臉部位置的列表
    # number_of_times_to_upsample 參數指定影像上採樣的次數，以幫助偵測小臉

    # 偵測臉部特徵點
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) 
                     for face_location in face_locations]  
    # 對每個偵測到的臉部位置，使用 pose_predictor_5_point 模型提取臉部的關鍵特徵點
    # 返回一個包含特徵點的列表，每個項目對應一張臉的特徵點

    # 編碼
    return [np.array(face_encoder.compute_face_descriptor(face_image, 
                                    raw_landmark_set, num_jitters)) for
                                    raw_landmark_set in raw_landmarks]  
    # 將每個臉部特徵點轉換為特徵向量，這些向量表示人臉的獨特特徵
    # compute_face_descriptor() 函數基於特徵點計算人臉的 128 維特徵向量
    # num_jitters 參數表示為了提高準確性，對特徵點進行擾動的次數

# 載入圖檔
known_image_1 = cv2.imread(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\jared_1.jpg") # cv2.imread() 讀取已知人臉的圖像
known_image_2 = cv2.imread(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\jared_2.jpg")  
known_image_3 = cv2.imread(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\jared_3.jpg")  
known_image_4 = cv2.imread(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\obama.jpg")  


unknown_image = cv2.imread(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_face\jared_4.jpg")  # 讀取要辨識的未知人臉圖像 jared_4.jpg

names = ["jared_1.jpg", "jared_2.jpg", "jared_3.jpg", "obama.jpg"]  # 用列表儲存已知圖像的名稱，這些名稱對應於前面載入的圖像

# 圖像編碼
known_image_1_encoding = face_encodings(known_image_1)[0]  # 將圖像轉換為臉部特徵向量
known_image_2_encoding = face_encodings(known_image_2)[0]  
known_image_3_encoding = face_encodings(known_image_3)[0]  
known_image_4_encoding = face_encodings(known_image_4)[0]  


known_encodings = [known_image_1_encoding, known_image_2_encoding, 
                   known_image_3_encoding, known_image_4_encoding]  
# 將所有已知圖像的臉部特徵向量儲存到列表 known_encodings 中

unknown_encoding = face_encodings(unknown_image)[0]  
# 將未知圖像 jared_4.jpg 的臉部特徵向量儲存在 unknown_encoding 變數中

# 比對
computed_distances = compare_faces(known_encodings, unknown_encoding)  
# 計算未知臉部與已知臉部之間的歐幾里得距離，結果是未知臉部與每個已知臉部的距離列表

computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, 
                                                      names, unknown_encoding)  
# 將這些距離與對應的名稱進行排序，返回排序後的距離和名稱

print('比較兩張臉的法向量距離：', computed_distances)  
# 輸出未知臉部與每個已知臉部之間的距離，越小的距離表示越相似

print('排序：', computed_distances_ordered)  
# 輸出排序後的距離列表

print('依相似度排序：', ordered_names)  
# 輸出根據相似度排序後的名稱列表，表示未知臉部與哪個已知臉部最相似


'''
