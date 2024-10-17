'''
1. 實例分割 (Instance Segmentation)   
    實例分割是一種計算機視覺技術，它不僅要對圖像中的物體進行分類，還需要將圖像中的每個物體與其周圍的背景精確地分開。
    具體來說，實例分割同時解決了「物體檢測」和「語義分割」兩個問題。
        1. 物體檢測：識別圖像中的不同物體，並用邊界框(Bounding Box)將它們框起來。
        2. 語義分割：對圖像中的每個像素進行分類，使得同一類別的像素被分為一組。
    實例分割的目標是在語義分割的基礎上，進一步區分圖像中同類的不同個體 (例如：同一圖像中的兩隻狗需要分別標註)

    
2. 發展背景 
    1.【無法精確區分物體的形狀】或【分割不同實例】：早期的目標檢測方法如 R-CNN、Fast R-CNN、Faster R-CNN 等技術在物體檢測上取得了很大進展，但只能生成物體的邊界框。
    2. 可以對每個像素進行分類，但【無法區分同一類別的不同個體】： 語義分割方法如 FCN
    3. 同時實現【物體檢測】和【像素級別的分割】：實例分割


3. Mask R-CNN 模型
    Mask R-CNN 是 Faster R-CNN 的擴展，通過在物體檢測的基礎上【增加一個分割分支】來實現實例分割，模型架構如下：

        1. 骨幹網絡 (Backbone Network)
            骨幹網絡主要用於【取圖像的特徵】。常用的骨幹網絡包括 ResNet 和 ResNeXt。
            這些網絡通過一系列的卷積層，提取多層次的特徵圖，這些特徵圖保留了圖像中的關鍵信息，如邊緣、顏色和形狀等。

            通常，在這些骨幹網絡中會加入 FPN (Feature Pyramid Network)，以增強對多尺度物體的檢測能力。
            FPN 允許在不同的尺度上提取特徵，使得網路能夠更好地處理大小不同的物體。

                【註】
                1. Feature Pyramid Network (FPN)
                    在傳統的 CNN 中，隨著網絡層數的增加，特徵圖的尺寸會逐漸減小，而特徵的抽象程度會提高。
                    意味著深層的特徵圖對於大物體具有較強的表現力，但對於小物體則表現較差。
                    反之，淺層的特徵圖雖然保留了更多的空間細節，適合檢測小物體，但其抽象能力較弱。

                    FPN 通過在不同層次的特徵圖之間建立金字塔結構，來整合多層次的特徵，從而增強對不同尺度物體的檢測能力。FPN 包括兩個關鍵步驟：
                    
                    1. 自上而下路徑：從深層特徵圖開始，逐步向上進行上採樣(編碼器)(通常通過最近鄰插值)。
                       在每個上採樣步驟中，將上採樣後的特徵與相應層次的淺層特徵進行融合（如通過逐元素相加）。
                    2. 橫向連接：這是用於將淺層的特徵圖與上採樣的深層特徵圖進行融合的一種操作。
                       這些橫向連接可以補充上採樣過程中可能丟失的空間信息，從而保留更多的細節。

                       
        2. RPN (Region Proposal Network)
            RPN 是一個輕量級的神經網絡，主要用於生成一系列的候選區域 (Regions of Interest, RoIs)。
            這些候選區域可能包含物體，並通過後續的處理進行分類和精確定位。
                
                1. 滑動窗口機制：在輸入的特徵圖上，使用一個固定大小的卷積核(通常是3x3)來滑動，這個卷積核用於檢測每個滑動窗口中的局部區域。
                2. 候選框(Anchors)生成：對於每個滑動窗口位置，RPN 會生成一組固定數量的候選框。這些 anchors 的大小和長寬比是預先設置的，通常會設置多種尺寸和比例以覆蓋不同尺度的物體。
                3. 候選框分類和回歸：對於每個 anchor，RPN 會生成兩個輸出：
                   (一) 二元分類：用於判斷該 anchor 是否包含物體 (稱為「物體/非物體」分類)
                   (二) 框回歸，用於調整 anchor 的位置和大小，使之更好地與真實物體匹配。
            
            
        3. RoIAlign
            RoIAlign 是 Mask R-CNN 引入的一種操作，用於解決 RoIPool 操作中存在的量化誤差問題。
                RoIPool 是 Faster R-CNN 中使用的一種方法，用於將任意大小的候選區域映射到固定大小的特徵圖的操作。
                RoIPool 使用了離散化 (即四捨五入) 操作，這會導致一些特徵信息的丟失，尤其是對於精細的實例分割任務，這種誤差會導致精度下降。           

            RoIAlign 通過取消量化操作，並採用雙線性插值來計算特徵點的值，從而保留了更多的空間信息。RoIAlign 包括以下步驟：
                1. 對齊：首先，對每個候選區域的坐標進行對齊操作，確保其與特徵圖上的位置精確匹配。
                2. 雙線性插值：對於對齊後的候選區域，使用雙線性插值來計算該區域內的每個網格點的特徵值。這避免了傳統 RoIPool 中的四捨五入操作所帶來的誤差。
                          
                    【註】
                    1. 雙線性插值 (Bilinear Interpolation)
                        用於在二維空間中估算未知點的值。它通過利用鄰近已知點的值，來計算一個位置(通常是非整數坐標點)的值。
                        雙線性插值的基本思想是在二維平面上使用線性插值的方法，首先沿一個方向進行線性插值，再沿另一個方向進行線性插值。
                        這樣就可以在一個網格(由四個已知點組成的矩形區域)內估算出任意位置的值。
                

                     
        4. 分類和邊界框回歸
            每個候選區域都會通過兩個不同的分支進行處理：一個分支用於對候選區域中的物體進行分類，另一個分支則用於對邊界框進行精確回歸 (調整其位置和尺寸)


        5. 掩碼分支 (Mask Branch)
            掩碼分支是 Mask R-CNN 的獨特之處。這個分支是一個小型的 FCN (全卷積網絡)，它對每個候選區域生成一個像素級的分割掩碼。
            掩碼的分辨率通常是 28x28 像素，並且對應於特定的分類標籤 (例如，假設候選區域被分類為「貓」，那麼這個分支就會生成一個「貓」的掩碼)。
            這個掩碼被後處理階段放大到與圖像大小一致，並且進行進一步的細化處理。

            【註】
                1. Fully Convolutional Network (FCN)
                    FCN 是一種卷積神經網絡架構，用於圖像分割任務。
                    傳統的卷積神經網絡在分類任務中通常包含卷積層和全連接層，卷積層負責提取特徵，全連接層負責最終分類。
                    然而，這種架構並不適合像素級的圖像分割任務，因為全連接層會丟失空間信息。
                    (由於全連接層要求輸入是扁平化的一維向量，因此在輸入全連接層之前，通常需要將來自卷積層或池化層的多維特徵圖壓縮成一維向量)                    
                    FCN 取消了全連接層，並將整個網絡改造成全卷積結構，這樣網絡能夠輸出與輸入圖像尺寸相同的特徵圖，從而實現像素級的分類。


'''

from PIL import Image  # PIL：圖像處理庫，Image：加載和處理圖像
import matplotlib.pyplot as plt  
import torch  
import torchvision.transforms as T  # transforms ：用於圖像的數據增強和預處理
import torchvision  # 許多用於計算機視覺的數據集、模型和圖像處理工具
import numpy as np  
import cv2  # OpenCV：電腦視覺庫，用於圖像處理
import random 
import time  
import os  


# LIST COCO_INSTANCE_CATEGORY_NAMES：映射COCO數據集的【分類名稱】
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)  
# 從 torchvision 加載一個預訓練的Mask R-CNN模型，該模型使用ResNet-50作為骨幹網絡並搭配FPN(Feature Pyramid Network)結構，pretrained=True：表示使用預訓練的權重
''' 
【註】ResNet (Residual Network) 由微軟研究院提出的一種深度卷積神經網絡，它解決了隨著網路層數增加而導致的梯度消失和模型退化問題。
      ResNet 通過引入殘差塊來緩解這個問題。
        Output = F(x) + x 
            1. x = 輸入
            2. F(x) = 卷積操作的結果 (通常包含兩層或三層卷積層)
            3. F(x) + x = 將輸入直接添加到卷積操作的結果上，即捷徑連接(Skip Connection)

【註】梯度消失是指在訓練深層神經網絡時，隨著梯度在反向傳播過程中逐層傳遞，梯度的數值會變得非常小，導致靠近輸入層的權重更新速度極慢。這會使得網絡難以有效學習。
      當梯度的數值非常小時，這意味著網絡中每層的權重更新幅度也會非常小。
         ΔW = -(η ∂L/ ∂W)
            1. ΔW = 權重更新值
            2. η = 學習率
            3. ∂L/ ∂W = 損失函數對權重的梯度 
'''

model  # print 模型的結構，方便查看模型的細節和參數


# 設定遮罩的顏色
def random_colour_masks(image):  
    # random_colour_masks 接受一個二維數組(圖像, (H, W))作為輸入，並返回一個彩色的遮罩圖像
    
    # 定義一個顏色列表，每個顏色用一個RGB值表示
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], 
               [255, 255, 0], [255, 0, 255], [80, 70, 180], [250, 80, 190],
               [245, 145, 50], [70, 150, 250], [50, 190, 190]]  
    
    # 創建三個與輸入圖像尺寸相同的空白圖像(r, g, b)，每個通道用於存儲對應顏色通道的值
    r = np.zeros_like(image).astype(np.uint8)  
    g = np.zeros_like(image).astype(np.uint8)  
    b = np.zeros_like(image).astype(np.uint8)  
    
    # 根據隨機選擇的顏色，將image中值為1的位置賦予相應的RGB值
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]  
    
    # 將r, g, b三個通道疊加在一起，形成彩色的遮罩
    coloured_mask = np.stack([r, g, b], axis=2)  
    
    # 返回生成的彩色遮罩
    return coloured_mask  




# 物件偵測，傳回遮罩、邊框、類別
def get_prediction(img_path, threshold): # get_prediction接受圖像路徑和分數閾值作為輸入，返回遮罩、邊框和類別標籤
    
    img = Image.open(img_path)  # Image.open()打開指定路徑的圖像
    
    transform = T.Compose([T.ToTensor()]) # 將圖像轉換為 tensor
    
    img = transform(img) # 圖像應用定義的transform (轉為tensor)
    
    pred = model([img]) # 將圖像傳入模型，進行預測。返回的pred包含了多個預測結果，如遮罩、邊框、類別標籤、分數等
    
    pred_score = list(pred[0]['scores'].detach().numpy()) # 獲取模型預測的分數列表，轉換為znumpy array，detach()將張量從計算圖中分離出來
    
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # 根據分數閾值，找到最後一個分數大於閾值的位置索引，用於篩選最可靠的預測結果
    
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy() # 獲取遮罩數據，將其轉換為二值圖像 (大於0.5的部分設為1)
    
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] 
                  for i in list(pred[0]['labels'].numpy())]  # 獲取預測的類別標籤，並通過映射得到對應的類別名稱
    
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] 
                  for i in list(pred[0]['boxes'].detach().numpy())]  # 獲取預測的邊框座標，並將其轉換為整數型座標，以便後續畫圖使用
    
    masks = masks[:pred_t + 1] # 只保留分數高於閾值的遮罩
    
    pred_boxes = pred_boxes[:pred_t + 1] # 只保留分數高於閾值的邊框
    
    pred_class = pred_class[:pred_t + 1] # 只保留分數高於閾值的類別標籤
    
    return masks, pred_boxes, pred_class # 返回篩選後的遮罩、邊框和類別標籤


# 物件偵測含遮罩上色、顯示結果
def instance_segmentation_api(img_path, threshold=0.5, rect_th=3, 
                              text_size=2, text_th=2):  # instance_segmentation_api進行物件偵測和實例分割，並顯示結果，參數包括圖像路徑、分數閾值、邊框厚度、文字大小和文字厚度等
    
    masks, boxes, pred_cls = get_prediction(img_path, threshold)  # 調用 get_prediction 獲取預測的遮罩、邊框和類別標籤
    
    img = cv2.imread(img_path) 
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 將圖像從BGR格式轉換為RGB格式，因為OpenCV默認讀取的是BGR格式
    
    for i in range(len(masks)):  
        rgb_mask = random_colour_masks(masks[i])  # 使用random_colour_masks 為每個遮罩生成隨機顏色的彩色遮罩
        
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0) # cv2.addWeighted：將彩色遮罩疊加在原圖上, 0.5：表示遮罩的透明度
        
        print(boxes[i][0], boxes[i][1]) # print 邊框的左上角和右下角座標
        
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), 
                      thickness=rect_th) # 在圖像上繪製矩形邊框，顏色為綠色，厚度由 rect_th 指定
        
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, # 在圖像上添加類別標籤的文字，位置為邊框左上角
                    text_size, (0, 255, 0), thickness=text_th) # 字體：HERSHEY_SIMPLEX, 顏色：綠色, 字體大小：text_size, 文字厚度(粗細)：text_th
    
    plt.figure(figsize=(20, 30)) # 創建一個新的圖形，指定顯示的大小
    
    plt.imshow(img)  
    plt.xticks([]) # 隱藏 x 軸刻度標籤
    plt.yticks([]) # 隱藏 y 軸刻度標籤
    plt.show()  



# 顯示測試圖檔
img = Image.open(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\Mask_RCNN\FudanPed00001.png')  
plt.imshow(img) 
plt.axis('off')  # 隱藏坐標軸
plt.show()  

# 模型預測
transform = T.Compose([T.ToTensor()])  # 轉換為 tensor
img_tensor = transform(img)  # 將圖像應用轉換

model.eval()  # 評估模式
pred = model([img_tensor])  # 使用模型，返回預測結果

pred[0]  # 打印出第一個預測的結果，包含遮罩、邊框、類別標籤和分數

# 保留遮罩值>0.5的像素，其他一律為 0
masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()  # 取得遮罩，將大於0.5的部分設為1，其他設為0，並轉換為numpy數組


# 顯示遮罩
plt.imshow(masks[0], cmap='gray')  # 顯示第一個遮罩，使用灰度圖表示
plt.axis('off')  
plt.show()  

# 遮罩上色
mask1 = random_colour_masks(masks[0])  # 使用自定義函數random_colour_masks為遮罩上色
plt.imshow(mask1)  
plt.axis('off')  
plt.show()  

# 原圖加遮罩
blend_img = cv2.addWeighted(np.asarray(img), 0.5, mask1, 0.5, 0)  # 將原圖與第一個彩色遮罩疊加，透明度各佔50%

# 第 2 個遮罩
mask2 = random_colour_masks(masks[1])  # random_colour_masks 為第二個遮罩上色
blend_img = cv2.addWeighted(np.asarray(blend_img), 0.5, mask2, 0.5, 0)  # 將疊加了第一個遮罩的圖像與第二個彩色遮罩再次疊加

plt.imshow(blend_img)  
plt.axis('off')  
plt.show()  

# 使用自定義API進行物件偵測和實例分割
instance_segmentation_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\Mask_RCNN\people1.jpg', 0.5, rect_th=1, 
                              text_size=1, text_th=1)  # 閾值0.5 過濾掉低於此值的預測結果，
                                                       # 矩形框(邊界框)的厚度為 1 個像素。當圖像中的物體實例被檢測出來後，邊界框會圍繞著這些實例繪製，表示出實例的位置。
                                                       # 文字大小為1，文字粗細為1

instance_segmentation_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\Mask_RCNN\car.jpg', 0.9, rect_th=5, text_size=2, text_th=2)  

instance_segmentation_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\Mask_RCNN\traffic.jpg', 0.6, rect_th=2, text_size=2, text_th=2)  

instance_segmentation_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\Mask_RCNN\birds.jpg', 0.9)  

instance_segmentation_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\Mask_RCNN\people2.jpg', 0.8, 
                          rect_th=1, text_size=1, text_th=1)  

instance_segmentation_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\Mask_RCNN\cat_dog.jpg', 0.95, rect_th=5, text_size=5, text_th=5)  

# 偵測所有物件，傳回遮罩
def pick_person_mask(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3): # pick_person_mask：輸入圖像路徑和閾值，返回人物遮罩
    masks, boxes, pred_cls = get_prediction(img_path, threshold)  # 獲取預測的遮罩、邊框和類別標籤
    person_ids = [i for i in range(len(pred_cls)) if pred_cls[i] == "person"]  # 篩選出所有類別標籤為"person"的索引
    person_masks = masks[person_ids, :, :]  # 根據索引提取人物的遮罩
    persons_mask = person_masks.sum(axis=0)  # 將所有人物的遮罩疊加為一個整體遮罩
    persons_mask = np.clip(persons_mask, 0, 1)  # 將遮罩的值限制在0到1之間，確保有效值
    return persons_mask  # 返回最終的人物遮罩


# 讀取檔案
img_path = r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\Mask_RCNN\blur.jpg"  # 指定要讀取的圖像檔案路徑
img = cv2.imread(img_path)  # OpenCV 讀取圖像，返回 numpy array 表示的圖像

# 取得人物遮罩
person_mask = pick_person_mask(img_path, threshold=0.5, rect_th=3,  # pick_person_mask： 獲取人物的遮罩
                               text_size=3, text_th=3).astype(np.uint8) # 遮罩轉換為8位無符號整數類型

# 把遮罩 RGB 設為相同值
person_mask = np.repeat(person_mask[:, :, None], 3, axis=2)  
# 將單通道的遮罩擴展為三通道，確保遮罩的每個通道(R、G、B)都相同，這樣可以方便後續與彩色圖像進行操作

# 照片模糊化
img_blur = cv2.GaussianBlur(img, (21, 21), 0)  
# 使用高斯模糊對整張圖片進行模糊處理，kernel大小為21x21，標準差為0

# 人物部分採用原圖，其他部分使用模糊化的圖
final_img = np.where(person_mask == 1, img, img_blur)  
# 使用 np.where 根據遮罩的值來選擇圖像的部分，當遮罩值為1時使用原圖，否則使用模糊後的圖像

# fix 中文亂碼 
from matplotlib.font_manager import FontProperties  # 設定字體屬性
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定字體為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 防止負號顯示錯誤

# 顯示原圖與生成圖，比較背景的處理效果
plt.figure(figsize=(15, 15))  # 創建一個新的圖形，大小為15x15英寸
plt.subplot(121)  # 在圖形中創建一個1行2列第1個子圖(左邊)
plt.title('原圖')  # 子圖標題
plt.imshow(img[:, :, ::-1])  # 將圖像的第三維度(顏色通道)反轉順序
plt.axis('off')  
plt.subplot(122)  # 創建第1行第2列二個子圖(右邊)
plt.title('生成圖')  # 第二個子圖標題
plt.imshow(final_img[:, :, ::-1])  
plt.axis('off')  
plt.show()  



