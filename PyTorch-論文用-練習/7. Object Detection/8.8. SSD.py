'''
SSD (Single Shot MultiBox Detector) 
它與YOLO(You Only Look Onc) 類似，都屬於單階段物件偵測方法。

SSD 演算法的核心是直接在不同尺度的特徵圖上同時進行物件位置回歸和類別分類。
這種方法相比兩階段物件偵測方法 (如FasterRCNN)，提高了物件偵測的速度和效率。

1. 基礎網路
    SSD通常使用一個預訓練的卷積神經網路(如VGG16)作為基礎網路，
    用於提取輸入圖像的特徵。這些特徵將用於後續的物件偵測。

2. 多尺度特徵圖(Multi-scale Feature Maps)
    SSD的創新點在於它使用多尺度特徵圖來進行物件偵測。
    這些特徵圖來自基礎網路的不同層，並且具有不同的尺寸和分辨率。
    每個特徵圖都會生成一組預設的錨框(Anchor Boxes)，並在這些錨框上進行物件位置回歸和類別分類。


SSD的工作流程

1. 輸入圖像處理
    將輸入圖像調整到固定尺寸 (如300x300)，並進行歸一化處理。
    
    1. 圖像調整尺寸
        為了適應神經網路的輸入要求，將所有輸入圖像調整到固定尺寸。
        例如，SSD300模型要求輸入圖像尺寸為300x300像素。這樣做可以使網路架構固定，便於處理不同大小的圖像。
   
     2. 歸一化處理：
        將圖像像素值縮放到0到1之間(通常是將像素值除以255)，這有助於加速模型的訓練並提高穩定性。


2. 特徵提取
    通過基礎網路提取圖像的特徵。

    1. 基礎網路：
        SSD通常使用預訓練的CNN作為基礎網路。這部分網路負責提取圖像的基本特徵，如邊緣、角點、顏色等。

    2. 特徵圖：
        基礎網路的不同層輸出不同分辨率的特徵圖，這些特徵圖包含了圖像中的多尺度信息。

          
3. 多尺度特徵圖生成：
    從基礎網路的不同層提取多尺度特徵圖，這些特徵圖具有不同的分辨率。

    1. 多尺度特徵圖：
        SSD在基礎網路的不同層提取特徵圖，這些特徵圖具有不同的分辨率。
        較低層的特徵圖具有高分辨率，適合偵測小物件；較高層的特徵圖具有低分辨率，適合偵測大物件。

    2. 擴展特徵圖：
        SSD還在基礎網路上增加了一些額外的卷積層，以進一步提取特徵圖。這些額外的特徵圖也具有不同的分辨率。


4. 錨框生成：
    在每個特徵圖上生成一組預設的錨框，這些錨框具有不同的比例和大小，以適應不同尺度和形狀的物件。

    1. 錨框 (Anchor Boxes)：
        在每個特徵圖的每個位置，SSD都會生成一組錨框。
        這些錨框的比例和大小是預先設置的，以適應不同形狀和大小的物件。

    2. 錨框比例和大小：
        SSD通常在每個特徵圖上生成幾個錨框，這些錨框的比例和大小會根據特徵圖的分辨率進行調整。
        例如，低分辨率的特徵圖可能會有較大的錨框，而高分辨率的特徵圖可能會有較小的錨框。

        
5. 位置回歸和類別分類：
    在每個錨框上同時進行位置回歸和類別分類，輸出每個錨框的位置信息和類別概率。


6. 非最大抑制(NMS)：
    過濾掉重疊的錨框，只保留置信度最高的錨框，最終得到物件偵測結果。
    
'''
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
"cuda" if torch.cuda.is_available() else "cpu"  

ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').to(device)  # 從 NVIDIA 的 torchhub 下載 預訓練的SSD模型
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')  # 從 NVIDIA 的 torchhub 載SSD處理工具
ssd_model.eval()  # 將模型設置為評估模式

classes_to_labels = utils.get_coco_object_dictionary()  # 獲取COCO數據集的類別字典
classes_to_labels  # 打印COCO類別字典

# 下載 3 張圖像
uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg',  
    'http://images.cocodataset.org/val2017/000000037777.jpg', 
    'http://images.cocodataset.org/val2017/000000252219.jpg'   
] 

# 轉為張量
inputs = [utils.prepare_input(uri) for uri in uris]  # 將每個圖像轉換為模型輸入格式
tensor = utils.prepare_tensor(inputs)  # 將所有圖像轉換為一個批次張量

# 預測
with torch.no_grad():  # no梯度計算
    detections_batch = ssd_model(tensor)  # 使用模型對批次圖像進行預測

# 篩選預測機率 > 0.4 的定界框
results_per_input = utils.decode_results(detections_batch)  # 解碼預測結果
best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]  # 篩選預測機率大於0.4的定界框

# 顯示結果
from matplotlib import pyplot as plt  
import matplotlib.patches as patches  # 從Matplotlib庫中導入patches模塊

for image_idx in range(len(best_results_per_input)):  # 遍歷每個圖像的結果
    fig, ax = plt.subplots(1)  # 創建圖像和軸
    # 顯示原圖
    image = inputs[image_idx] / 2 + 0.5  # 將圖像轉換回原始範圍
    ax.imshow(image)  # 顯示圖像
    
    # 顯示偵測結果
    bboxes, classes, confidences = best_results_per_input[image_idx]  # 獲取定界框、類別和置信度
    for idx in range(len(bboxes)):  # 遍歷每個定界框
        left, bot, right, top = bboxes[idx]  # 獲取定界框的左、下、右、上的坐標
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]  # 調整定界框的尺寸
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')  # 創建矩形補丁
        ax.add_patch(rect)  # 在軸上添加矩形補丁
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))  # 添加文字標籤
plt.show() 
