'''
FasterRCNN 在 RCNN 和 FastRCNN 的基礎上進行了改進，提高了目標檢測的速度和精度。
FasterRCNN 通過引入區域建議網絡 Region Proposal Network, RPN ，實現了端到端的訓練和預測。

1. 特徵提取網路

    FasterRCNN使用卷積神經網絡 (通常是VGG或ResNet) 來提取圖像特徵。
    這部分網絡接受原始圖像作為輸入，通過多層卷積操作生成高層特徵圖 (Feature map) 。

        1. 卷積層：提取局部特徵。
        2. 池化層：減少特徵圖的尺寸，同時保留重要信息。
        3. 最終輸出：高層次的特徵圖，包含圖像中的豐富信息。

2. 區域提案網路 Region Proposal Network, RPN

    RPN 是 FasterR-CNN 核心創新部分，負責從特徵圖中生成候選區域。
    它在特徵圖的每個位置生成一組錨框(anchor boxes)，每個錨框有不同的大小和比例。

        1. 在特徵圖的每個滑動窗口位置，生成一組錨框 
        2. 每個錨框都會有兩個輸出：一個是該錨框是否包含物體的概率（分類），另一個是該錨框相對於真實目標的回歸偏移量（回歸）。
        3. 根據分類結果，篩選出高概率的候選區域，並應用回歸偏移量進行調整，生成最終的區域提案。

            1. 錨框生成：在每個特徵圖的位置生成一組固定大小和比例的錨框。
            2. 分類與回歸：RPN 對每個錨框進行二分類（是否包含物體）和回歸（修正錨框位置）。
            3. 候選區域篩選：根據分類結果篩選出好的的區域提案，並應用回歸結果進行位置修正。

            
3. RoI池化層 RoI Pooling Layer

    RoI池化層將RPN生成的候選區域映射回特徵圖，並將每個候選區域轉換為固定大小的特徵圖。
    這樣，後續的全連接層可以處理固定大小的輸入。

        1. 將每個區域提案映射回原始特徵圖。
        2. 將映射的區域劃分為固定數量的子區域 (如7x7) ，並對每個子區域進行最大池化操作。
        3. 最終輸出固定大小的特徵圖，這些特徵圖可以用作後續分類和回歸的輸入。


4. 分類與回歸網路 Classification and Regression Network 

    這部分在RoI池化層生成的固定大小特徵圖上應用全連接層，進行進一步的分類和回歸。

        1. 分類：根據每個RoI特徵圖，判斷該區域所屬的類別，包括背景類別。
        2. 回歸：對每個RoI進行邊界框的回歸，細化區域提案的邊界，使其更加準確地包圍目標物體。

            1. 全連接層：將固定大小的特徵圖展平，並通過幾層全連接層進行特徵提取。
            2. 分類層：輸出每個RoI的分類結果，包含所有可能的類別。
            3. 回歸層：輸出每個RoI的邊界框偏移量，用於精確調整提案邊界。

            
5. 非極大值抑制 Non-Maximum Suppression, NMS

    過濾掉冗餘的邊界框，只保留最有可能的目標框。

'''

from PIL import Image 
import matplotlib.pyplot as plt  
import torch  
import torchvision.transforms as T 
import torchvision  
import numpy as np  
import cv2  
import os  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
"cuda" if torch.cuda.is_available() else "cpu"  

device = "cpu"  

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)  # 預訓練 FasterRCNN 模型
model.eval()  

COCO_INSTANCE_CATEGORY_NAMES = [  # 定義COCO數據集中的所有類別名稱
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

len(COCO_INSTANCE_CATEGORY_NAMES)  # 獲取COCO類別名稱的數量

def get_prediction(img_path, threshold): 
    img = Image.open(img_path)  # 讀取圖像
    transform = T.Compose([T.ToTensor()])  # 定義圖像轉換為tensor變換
    img = transform(img)  # 將圖像轉換為張量
    pred = model([img])  # 使用模型進行預測
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]  # 獲取預測的類別名稱
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]  # 獲取預測的定界框
    pred_score = list(pred[0]['scores'].detach().numpy())  # 獲取預測的分數
    
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  # 獲取超過門檻值的框的索引
    pred_boxes = pred_boxes[:pred_t+1]  # 過濾出超過門檻值的框
    pred_class = pred_class[:pred_t+1]  # 過濾出超過門檻值的類別
    return pred_boxes, pred_class  # 返回定界框和類別

def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=2, text_th=2):  # 定義物體檢測的API
    boxes, pred_cls = get_prediction(img_path, threshold)  # 獲取預測結果
    
    img = cv2.imread(img_path)  # 讀取圖像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將圖像從BGR轉換為RGB
    for i in range(len(boxes)):  # 遍歷所有定界框
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)  # 在圖像上畫框
        cv2.putText(img, pred_cls[i], (boxes[i][0][0], boxes[i][0][1]-10),  # 在圖像上寫文字
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    text_size, (0, 255, 0), thickness=text_th)
    plt.figure(figsize=(20, 30))  # 設置顯示圖像的大小
    plt.imshow(img)  # 顯示圖像
    plt.xticks([])  # 隱藏x軸刻度
    plt.yticks([])  # 隱藏y軸刻度
    plt.show()  # 顯示圖像

# 呼叫物件偵測的 API
object_detection_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_Object_Detection\people.jpg', threshold=0.8)  # 檢測圖像中的物體

# 呼叫物件偵測的 API
object_detection_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_Object_Detection\car.jpg', threshold=0.8)  # 檢測圖像中的物體

# 呼叫物件偵測的 API
object_detection_api(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_Object_Detection\traffic_scene.jpg', threshold=0.8, rect_th=1, text_size=1, text_th=1)  # 檢測圖像中的物體
