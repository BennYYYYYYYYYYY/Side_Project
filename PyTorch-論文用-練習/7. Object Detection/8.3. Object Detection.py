'''
若要理解這個物體檢測模型的運作概念，您需要掌握以下幾個數學概念和理論基礎：


1. 影像金字塔 (Image Pyramid)

    影像金字塔是多尺度圖像的一種表示方法，用於在不同尺度上進行物體檢測。
    每一層的圖像是上一層圖像縮小後的版本，這樣可以檢測不同大小的物體。

    對於給定的影像，縮放尺度s為：
    s = s_0 * scale 
    其中 s_0 是原始影像的尺寸，scale 是縮放比例。


2. 滑動窗口 (Sliding Window)

    滑動窗口是一種遍歷圖像的技術，將一個固定大小的窗口在圖像上滑動，並在每個位置提取子圖像（ROI, Region of Interest）。這些子圖像作為模型的輸入進行分類。

    滑動窗口的步長(stride) 決定了窗口每次移動的距離。假設窗口大小為 w*h，步長為 `stride`，那麼窗口的位置為：
    (x, y) = (i*stride ,j*stride )
    其中 i, j 為整數索引。

3. 非極大值抑制 (Non-Maximum Suppression, NMS)

    NMS是一種後處理技術，用於去除重疊的檢測框，僅保留最高置信度的檢測結果。

    重疊率 (Intersection over Union, IoU)： 用於衡量兩個檢測框的重疊程度。
    IoU: Area of Intersection / Area of Union

    NMS步驟：
    1. 根據置信度對所有檢測框排序。
    2. 選擇置信度最高的檢測框，將其加入最終檢測框集合。
    3. 將其餘與當前框的IoU大於閾值的檢測框剔除。
    4. 重複上述步驟，直到沒有剩餘的檢測框。

'''

import torch  
import torch.nn as nn  # 神經網路模塊
import torch.optim as optim  
from torch.optim import lr_scheduler  # 學習率調度器
import torchvision  # 用於計算機視覺任務
from torchvision import datasets, models, transforms 
import numpy as np  
import time  
import cv2  # 導入OpenCV，用於圖像處理

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
"cuda" if torch.cuda.is_available() else "cpu"  
device = "cpu"  

# 參數設定
WIDTH = 600  # 圖像縮放為 (600, 600)
PYR_SCALE = 1.5  # 影像金字塔縮放比例
WIN_STEP = 16  # 視窗滑動步數
ROI_SIZE = (250, 250)  # 視窗大小
INPUT_SIZE = (224, 224)  # CNN的輸入尺寸

model = models.resnet50(pretrained=True).to(device)  # 加載預訓練的ResNet50模型，並將其移動到指定的設備

from PIL import Image  

filename = r'PyTorch/Pytorch data/images_Object_Detection/bike.jpg'  # 設置圖像文件的路徑
orig = Image.open(filename)  # 打開圖像文件並加載為PIL圖像
orig = orig.resize((WIDTH, int(orig.size[1] / orig.size[0] * WIDTH)))  # 等比例縮放圖像，使其寬度為600像素
Width_Height_ratio = orig.size[1] / orig.size[0]  # 計算圖像的寬高比
orig.size  # 獲取圖像的大小

# 滑動視窗函數        
def sliding_window(image, step, ws):
    for y in range(0, image.size[1] - ws[1], step):  # 向下滑動 stepSize 格
        for x in range(0, image.size[0] - ws[0], step):  # 向右滑動 stepSize 格
            yield (x, y, image.crop((x, y, x + ws[0], y + ws[1])))  # 傳回裁剪後的視窗

# 影像金字塔函數
# image：原圖
# scale：每次縮小倍數
# minSize：最小尺寸
def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    yield image  # 第一次傳回原圖
    while True:
        w = int(image.size[0] / scale)  # 計算縮小後的尺寸
        image = image.resize((w, int(Width_Height_ratio * w)))  # 縮小圖像
        if image.size[0] < minSize[1] or image.size[1] < minSize[0]:  # 直到最小尺寸為止
            break
        yield image  # 傳回縮小後的圖像

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
    open_cv_image = np.array(pil_image)  # 將PIL圖像轉為numpy數組
    return open_cv_image[:, :, ::-1].copy()  # 將RGB格式轉換為BGR格式

# 產生影像金字塔
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)  # 生成圖像的金字塔
rois = torch.tensor([])  # 初始化存儲候選框的tensor
locs = []  
for image in pyramid:
    scale = WIDTH / float(image.size[0])  # 計算縮放比例
    print(image.size, 1/scale)  # print 圖像大小和縮放比例
    
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):  # 遍歷滑動窗口
        x = int(x * scale)  # 計算原始圖像中窗口的x坐標
        y = int(y * scale)  # 計算原始圖像中窗口的y坐標
        w = int(ROI_SIZE[0] * scale)  # 計算原始圖像中窗口的寬度
        h = int(ROI_SIZE[1] * scale)  # 計算原始圖像中窗口的高度

        roi = transform(roiOrig)  # 將窗口內的圖像塊轉換為模型輸入格式
        roi = roi.unsqueeze(0)  # 增加一個批次維度

        if len(rois.shape) == 1:  # 如果rois張量是空的
            rois = roi  # 初始化rois張量
        else:
            rois = torch.cat((rois, roi), dim=0)  # 將新候選框添加到rois張量中
        locs.append((x, y, x + w, y + h))  # 將窗口位置添加到位置列表中

rois = rois.to(device)  

print(locs)  # print 所有窗口的位置

rois.shape  # 候選框張量的形狀

# 讀取類別列表
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]  # 文件中讀取ImageNet類別，去除每行的空白字符

# 預測
model.eval()  
with torch.no_grad():  
    output = model(rois)  # 對候選框進行預測

# 轉成機率
probabilities = torch.nn.functional.softmax(output, dim=1)  # 將輸出轉換為機率分佈

# 取得第一名
top_prob, top_catid = torch.topk(probabilities, 1)  # 取得每個候選框的最高概率和對應的類別
probabilities  # 顯示所有候選框的概率分佈

probabilities[0, 518]  # 獲取第0個候選框的第518類的概率

torch.topk(probabilities[202], 1)  # 獲取第202個候選框的最高概率和對應的類別

top_catid.numpy().reshape(-1)  # 將top_catid轉換為NumPy數組並展平

for i in range(probabilities.shape[0]):  # 遍歷所有候選框
    print(i, probabilities[i, 671].item())  # 打印每個候選框的第671類的概率值

MIN_CONFIDENCE = 0.4  # 辨識機率門檻值

labels = {}
for (i, p) in enumerate(zip(top_prob.numpy().reshape(-1),  # 將top_prob轉為numpy數組並攤平
                            top_catid.numpy().reshape(-1))):  # 將top_catid轉為numpy數組並攤平
    (prob, imagenetID) = p  # 解壓每個候選框的概率和ImageNetID
    label = categories[imagenetID]  # 根據ImageNetID獲取類別標籤

    # 機率大於設定值，則放入候選名單
    if prob >= MIN_CONFIDENCE:
        # 只偵測自行車(671)
        if imagenetID != 671: continue  # 如果類別不是自行車，則跳過
        # 放入候選名單
        box = locs[i]  # 獲取候選框的位置
        L = labels.get(label, [])  # 獲取當前類別的候選框列表，如果不存在則創建新列表
        L.append((box, prob))  # 將候選框和概率添加到列表中
        labels[label] = L  # 更新字典中的類別和對應的候選框列表

labels.keys()  # 獲取所有類別的鍵

labels['mountain bike']  # 獲取mountain bike類別的所有候選框

# https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
def nms_pytorch(P ,thresh_iou):  # 定義非極大值抑制函數
    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]  # 獲取每個候選框的左上角x坐標
    y1 = P[:, 1]  # 左上角y坐標
    x2 = P[:, 2]  # 右下角x坐標
    y2 = P[:, 3]  # 右下角y坐標

    # we extract the confidence scores as well
    scores = P[:, 4]  # 獲取每個候選框的置信度分數

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)  # 計算每個候選框的面積
    
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()  # 根據置信度分數對候選框進行排序

    # initialise an empty list for 
    # filtered prediction boxes
    keep = []  # 初始化存儲過濾後候選框的列表
    

    while len(order) > 0:  # 當排序列表中還有元素時
        
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]  # 獲取置信度最高的候選框的索引

        # push S in filtered predictions list
        keep.append(P[idx])  # 將該候選框添加到過濾後的候選框列表中

        # remove S from P
        order = order[:-1]  # 從排序列表中移除該候選框

        # sanity check
        if len(order) == 0:  # 如果排序列表為空，則退出循環
            break
        
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)  # 根據排序索引獲取其餘候選框的左上角x坐標
        xx2 = torch.index_select(x2, dim=0, index=order)  # 右下角x坐標
        yy1 = torch.index_select(y1, dim=0, index=order)  # 左上角y坐標
        yy2 = torch.index_select(y2, dim=0, index=order)  # 右下角y坐標

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])  # 計算交叉區域的左上角x坐標
        yy1 = torch.max(yy1, y1[idx])  # 左上角y坐標
        xx2 = torch.min(xx2, x2[idx])  # 右下角x坐標
        yy2 = torch.min(yy2, y2[idx])  # 右下角y坐標

        # find height and width of the intersection boxes
        w = xx2 - xx1  # 計算交叉區域的寬度
        h = yy2 - yy1  # 計算高度
        
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)  # 避免非重疊框造成的寬度
        h = torch.clamp(h, min=0.0)  # 高度

        # find the intersection area
        inter = w * h  # 計算交叉區域的面積

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)  # 根據排序索引獲取其餘候選框的面積

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]  # 計算每個候選框與當前候選框S的並集

        # find the IoU of every prediction in P with S
        IoU = inter / union  # 計算每個候選框與當前候選框S的IoU值

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou  # 保留IoU小於閾值的候選框
        order = order[mask]  # 根據遮罩更新排序列表
    
    return keep  # 返回過濾後的候選框列表

def non_max_suppression_slow(boxes, overlapThresh=0.5):
    if len(boxes) == 0:  # 如果候選框數量為0，則返回空列表
        return []
    
    pick = []  # 儲存篩選的結果
    x1 = boxes[:,0]  # 取得候選視窗的左x座標
    y1 = boxes[:,1]  # 上y座標
    x2 = boxes[:,2]  # 右x座標
    y2 = boxes[:,3]  # 下y座標
    
    # 計算候選視窗的面積
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)  # 依視窗的底y座標排序
    
    # 比對重疊比例
    while len(idxs) > 0:  # 當索引列表不為空時
        last = len(idxs) - 1  # 取得最後一個索引
        i = idxs[last]  # 取得最後一個候選框的索引
        pick.append(i)  # 添加到列表
        suppress = [last]  # 初始化列表
        
        # 比對最後一筆與其他視窗重疊的比例
        for pos in range(0, last):
            j = idxs[pos]  # 取得其他候選框的索引
            
            # 取得所有視窗的涵蓋範圍
            xx1 = max(x1[i], x1[j])  # 計算交集區域的左x座標
            yy1 = max(y1[i], y1[j])  # 上y座標
            xx2 = min(x2[i], x2[j])  # 右x座標
            yy2 = min(y2[i], y2[j])  # 下y座標
            w = max(0, xx2 - xx1 + 1)  # 計算交集的寬度
            h = max(0, yy2 - yy1 + 1)  # 計算交集的高度
            
            # 計算重疊比例
            overlap = float(w * h) / area[j]
            
            # 如果大於門檻值，則儲存起來
            if overlap > overlapThresh:
                suppress.append(pos)  # 將索引添加到列表中
                
        # 刪除合格的視窗，繼續比對
        idxs = np.delete(idxs, suppress)  # 刪除已處理的索引
        
    # 傳回合格的視窗
    return boxes[pick]  # 返回篩選後的候選框

# 掃描每一個類別
for label in labels.keys():
    # if label != categories[671]: continue  # 如果類別不是自行車，則跳過
    
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
    proba = np.array([p[1] for p in labels[label]])  # 獲取所有候選框的機率
    # print(boxes.shape, proba.shape)
    # boxes = nms_pytorch(torch.cat((torch.tensor(boxes), 
    #    torch.tensor(proba).reshape(proba.shape[0], -1)), dim=1) , 
    #    MIN_CONFIDENCE) # non max suppression
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
