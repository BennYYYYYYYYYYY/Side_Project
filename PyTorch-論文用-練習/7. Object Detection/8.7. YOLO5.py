'''
YOLO (You Only Look Once)
YOLO 系列演算法的核心思想是將整張圖片一次性地通過神經網路，直接預測出所有可能的物件邊界框及其類別機率。
這種方法相比傳統的滑動視窗方法和區域提議網路(RPN)，提高了物件偵測的速度和效率。
網路架構可以分為三個主要部分: Backbone、Neck 和 Head

1. Backbone

Backbone 用於從輸入圖像中提取豐富的特徵。YOLOv5通常使用 CSPDarknet5 3作為其 Backbone。
CSP (Cross Stage Partial) Net是一種改進的 ResNet 架構，
它通過部分特徵融合和剩餘連接來減少計算量，同時保留了特徵表示能力。

        1. 特徵提取：通過多層卷積和池化操作，逐步縮小特徵圖的尺寸，增強特徵表示。
        2. CSP結構：通過部分殘差塊來進一步優化計算效率和模型性能。

            
2. Neck 
Neck的主要目的是生成多尺度的特徵圖，這對於偵測不同大小的物件非常重要。
YOLOv5 使用 FPN(Feature Pyramid Network) 和 PAN (Path Aggregation Network)來實現這一功能。

        1. FPN：通過自頂向下的路徑逐步融合來自不同層的特徵，生成不同尺度的特徵圖。
        2. PAN：通過自底向上的路徑進一步增強特徵融合，強化不同尺度特徵之間的信息流動。

        
3. Head
Head負責最終的邊界框回歸和物件分類。
YOLOv5的Head結構由多個卷積層組成，最終輸出的是每個網格單元的多個邊界框及其對應的類別機率。

        1. 邊界框回歸：輸出每個邊界框的中心坐標、寬度、高度。
        2. 物件分類：輸出每個邊界框所屬的類別機率。
'''
import torch  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
"cuda" if torch.cuda.is_available() else "cpu"  

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)  # 從 Ultralytics 的 YOLOv5庫 加載預訓練的 YOLOv5s 模型

# 批次處理
imgs = ['https://ultralytics.com/images/zidane.jpg',  # 圖片網址
        r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_Object_Detection\car.jpg']  # 本地圖片路徑

# 預測
results = model(imgs)  

# 輸出結果
results.print()  # 預測結果

results.save()  # 保存預測結果

results.show()  # 顯示結果

from IPython.display import Image
Image('./runs/detect/exp/zidane.jpg')

results.xyxy[0] 
results.pandas().xyxy[0]