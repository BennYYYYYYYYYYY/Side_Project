'''

1. 圖像分割
    Selective Search首先對圖像進行超像素分割。
    超像素分割將圖像分割成一系列更小且均勻的區域（超像素）。這些超像素有助於減少處理的計算量，同時保持圖像的語義結構。

2. 區域合併

    Selective Search基於多種特徵 (如顏色、紋理、大小和填充度)逐步合併相鄰的超像素，
    生成更大、更具語義的區域。這些區域在不同的尺度上提供了可能包含目標的候選區域。

3. 生成候選區域

    通過區域合併，Selective Search生成多個候選區域，這些區域可能包含潛在的目標物體。
    這些候選區域稱為區域提案 (Region Proposals)。
    這些提案覆蓋了圖像中的所有潛在目標，從而為後續的分類和回歸步驟提供了候選框。

4. 特徵提取

    在獲得候選區域後，下一步是從這些區域中提取特徵。
    這通常使用CNN進行特徵提取。CNN將每個候選區域作為輸入，並生成一組高維度的特徵向量，這些特徵向量描述了區域內的視覺特徵。

5. 分類與回歸

    提取特徵後，需要對每個候選區域進行分類和回歸。

    1. 分類：判斷每個候選區域是否包含物體，以及物體的類別。
    2. 回歸：細化候選區域的邊界框，使其更加準確地包圍目標物體。

6. 非極大值抑制 NMS

    在進行分類和回歸後，可能會得到多個重疊的邊界框。
    非極大值抑制是一種後處理步驟，用於過濾掉冗餘的邊界框，只保留最有可能的目標框。

'''
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.optim import lr_scheduler  # 學習率調度器
import torchvision  # 視覺任務
from torchvision import datasets, models, transforms 
import numpy as np  
import time  
import cv2  # 圖像處理

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
"cuda" if torch.cuda.is_available() else "cpu" 
device = "cpu"  

# 參數設定
WIDTH = 600  # 圖像縮放為 (600, 600)
INPUT_SIZE = (224, 224)  # CNN的輸入尺寸

model = models.resnet50(pretrained=True).to(device)  # 加載預訓練的ResNet50模型

from PIL import Image  

filename = r'PyTorch/Pytorch data/images_Object_Detection/bike.jpg'  
orig = Image.open(filename)  # 打開圖像文件並加載為PIL圖像
# 等比例縮放圖片
orig = orig.resize((WIDTH, int(orig.size[1] / orig.size[0] * WIDTH)))  # 等比例縮放圖像，使其寬度為600像素
Width_Height_ratio = orig.size[1] / orig.size[0]  # 計算圖像的寬高比
orig.size  # 獲取圖像的大小

# 轉換函數
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),  # 調整圖像大小為 (224, 224)
    transforms.ToTensor(),  # 將圖像轉換為張量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 標準化圖像
                         std=[0.229, 0.224, 0.225])
])

# PIL格式轉換為OpenCV格式
def PIL2CV2(orig):
    pil_image = orig.copy()  # 複製PIL圖像
    open_cv_image = np.array(pil_image)  
    return open_cv_image[:, :, ::-1].copy()  # 將RGB格式轉換為BGR格式

# 產生 Selective Search 影像
import matplotlib.pyplot as plt 

plt.figure(figsize=(16, 16))  # 設置圖像顯示的大小
def Selective_Search(img_path):
    img = cv2.imread(img_path)  # 讀取圖像
    img = cv2.resize(img, (WIDTH, int(orig.size[1] / orig.size[0] * WIDTH)), interpolation=cv2.INTER_AREA)  # 調整圖像大小
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將圖像從BGR格式轉換為RGB格式
    
    # 執行 Selective Search
    cv2.setUseOptimized(True)  # 啟用OpenCV優化
    cv2.setNumThreads(8)  # 設置線程數量
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # 創建Selective Search對象
    gs.setBaseImage(img)  # 設置基礎圖像
    gs.switchToSelectiveSearchFast()  # 使用快速模式的Selective Search
    rects = gs.process()  # 獲取候選框
    # print(rects)
    
    rois = torch.tensor([])  # 初始化存儲候選框的張量
    locs = []  # 初始化位置列表
    j = 1  # 用於顯示圖像的計數器
    for i in range(len(rects)):  # 遍歷所有候選框
        x, y, w, h = rects[i]  # 獲取候選框的位置和大小
        if w < 100 or w > 400 or h < 100: continue  # 過濾掉寬度小於100或大於400或高度小於100的候選框
            
        # 框與原圖的比例
        scale = WIDTH / float(w)

        # 縮放圖形以符合模型輸入規格 
        crop_img = img[y:y+h, x:x+w]  # 裁剪圖像
        crop_img = Image.fromarray(crop_img)  # 將NumPy數組轉換為PIL圖像
        if j <= 100:  # 顯示前100個候選框
            plt.subplot(10, 10, j)  # 創建子圖
            plt.imshow(crop_img)  # 顯示圖像
        j += 1  
        
        roi = transform(crop_img)  # 將圖像轉換為模型輸入格式
        roi = roi.unsqueeze(0)  # 增加一個批次維度

        # 加入輸出變數中
        if len(rois.shape) == 1:  # 如果rois張量是空的
            rois = roi  # 初始化rois張量
        else:
            rois = torch.cat((rois, roi), dim=0)  # 將新候選框添加到rois張量中
        locs.append((x, y, x + w, y + h))  # 將候選框的位置添加到位置列表中

    return rois.to(device), locs  # 返回候選框張量和位置列表

rois, locs = Selective_Search(filename)  # 執行Selective Search並獲取候選框和位置
plt.tight_layout()  # 調整子圖布局

rois.shape  

len(locs)  

locs  


# 讀取類別列表
with open("imagenet_classes.txt", "r") as f:  # 打開ImageNet類別文件
    categories = [s.strip() for s in f.readlines()]  # 讀取文件中的每一行並去除換行符，存儲在categories列表中

# 預測
model.eval()  # 將模型設置為評估模式
with torch.no_grad():  # 禁用梯度計算
    output = model(rois)  # 使用模型對候選框進行預測
    
# 轉成機率
probabilities = torch.nn.functional.softmax(output, dim=1)  # 將模型輸出轉換為機率分佈

# 取得第一名
top_prob, top_catid = torch.topk(probabilities, 1)  # 取得每個候選框的最高概率和對應的類別
probabilities  # 顯示所有候選框的概率分佈

top_catid.numpy().reshape(-1)  # 將top_catid轉換為NumPy數組並展平

for i in range(probabilities.shape[0]):  # 遍歷所有候選框
    print(i, probabilities[i, 671].item())  # 打印每個候選框的第671類的概率值

probabilities[0, 671]  # 獲取第0個候選框的第671類的概率

MIN_CONFIDENCE = 0.4  # 辨識機率門檻值

labels = {}  # 初始化存儲結果的字典
for (i, p) in enumerate(zip(top_prob.numpy().reshape(-1),  # 將top_prob轉換為NumPy數組並展平
                            top_catid.numpy().reshape(-1))):  # 將top_catid轉換為NumPy數組並展平
    (prob, imagenetID) = p  # 解壓每個候選框的概率和ImageNetID
    label = categories[imagenetID]  # 根據ImageNetID獲取類別標籤

    # 機率大於設定值，則放入候選名單
    if prob >= MIN_CONFIDENCE:
        # 只偵測自行車(671)
        if imagenetID != 671: continue  # 如果類別不是自行車，跳過
        # 放入候選名單
        box = locs[i]  # 獲取候選框的位置
        print(i, imagenetID)  # 打印候選框的索引和ImageNetID
        L = labels.get(label, [])  # 獲取當前類別的候選框列表，如果不存在則創建新列表
        L.append((box, prob))  # 將候選框和概率添加到列表中
        labels[label] = L  # 更新字典中的類別和對應的候選框列表

labels.keys()  # 獲取所有類別的鍵

labels['mountain bike']  # 獲取'mountain bike'類別的所有候選框

def non_max_suppression_slow(boxes, overlapThresh=0.5):  
    if len(boxes) == 0:  # 如果候選框數量為0，則返回空列表
        return []
    
    pick = []  # 儲存篩選的結果
    x1 = boxes[:, 0]  # 左x座標
    y1 = boxes[:, 1]  # 上y座標
    x2 = boxes[:, 2]  # 右x座標
    y2 = boxes[:, 3]  # 下y座標
    
    # 計算候選視窗的面積
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)  # 依視窗的底y座標排序
    
    # 比對重疊比例
    while len(idxs) > 0:  # 當索引列表不為空時
        last = len(idxs) - 1  # 取得最後一個索引
        i = idxs[last]  # 取得最後一個候選框的索引
        pick.append(i)  # 將該索引添加到結果列表中
        suppress = [last]  # 初始化抑制列表
        
        # 比對最後一筆與其他視窗重疊的比例
        for pos in range(0, last):
            j = idxs[pos]  # 取得其他候選框的索引
            
            # 取得所有視窗的涵蓋範圍
            xx1 = max(x1[i], x1[j])  # 計算交集的左x座標
            yy1 = max(y1[i], y1[j])  # 計算交集的上y座標
            xx2 = min(x2[i], x2[j])  # 計算交集的右x座標
            yy2 = min(y2[i], y2[j])  # 計算交集的下y座標
            w = max(0, xx2 - xx1 + 1)  # 計算交集的寬度
            h = max(0, yy2 - yy1 + 1)  # 計算交集的高度
            
            # 計算重疊比例
            overlap = float(w * h) / area[j]
            
            # 如果大於門檻值，則儲存起來
            if overlap > overlapThresh:
                suppress.append(pos)  # 將索引添加到抑制列表中
                
        # 刪除合格的視窗，繼續比對
        idxs = np.delete(idxs, suppress)  # 刪除已處理的索引
        
    # 傳回合格的視窗
    return boxes[pick]  # 返回篩選後的候選框

# 掃描每一個類別
for label in labels.keys():
    # if label != categories[671]: continue  
    
    # 複製原圖
    open_cv_image = PIL2CV2(orig)  # 將PIL圖像轉換為OpenCV格式

    # 畫框
    for (box, prob) in labels[label]:  # 遍歷當前類別的所有候選框
        (startX, startY, endX, endY) = box  # 獲取候選框的位置
        cv2.rectangle(open_cv_image, (startX, startY), (endX, endY), (0, 255, 0), 2)  # 在原圖上畫框

    # 顯示 NMS(non-maxima suppression) 前的框
    cv2.imshow("Before NMS", open_cv_image)  # 顯示原圖

    # NMS
    open_cv_image2 = PIL2CV2(orig)  # 將PIL圖像轉換為OpenCV格式
    boxes = np.array([p[0] for p in labels[label]])  # 獲取所有候選框的位置
    proba = np.array([p[1] for p in labels[label]])  # 獲取所有候選框的概率
    boxes = non_max_suppression_slow(boxes, MIN_CONFIDENCE)  # 使用非極大值抑制來過濾候選框
    
    color_list = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 0), (0, 255, 255)]  # 定義顏色列表
    for i, x in enumerate(boxes):  # 遍歷過濾後的候選框
        # startX, startY, endX, endY, label = x.numpy()
        startX, startY, endX, endY = x  # 獲取候選框的位置
        # 畫框及類別
        cv2.rectangle(open_cv_image2, (int(startX), int(startY)), (int(endX), int(endY)), color_list[i % len(color_list)], 2)  # 在圖像上畫框
        startY = startY - 15 if startY - 15 > 0 else startY + 15  # 計算文字標籤的位置
        cv2.putText(open_cv_image2, str(label), (int(startX), int(startY)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)  # 在圖像上畫文字標籤

    # 顯示
    cv2.imshow("After NMS", open_cv_image2)  # 顯示過濾後的圖像
    cv2.waitKey(0)  # 等待按鍵輸入
            
cv2.destroyAllWindows()  # 關閉所有視窗

boxes  # 返回過濾後的候選框
