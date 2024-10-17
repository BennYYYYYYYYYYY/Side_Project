'''
1. 風格轉換(Style Transfer)
    
    在風格轉換中，目標是創造一張新圖像，這張圖像的「內容一致」，但「風格不同」。
    風格轉換依賴於CNN中間層的激活來表示圖像的內容和風格。主要作法是重新定義損失函數，分為內容損失、風格損失。

        1. 內容圖像 (Content Image)
            想保留下的【主要形狀或結構的圖像】。
                例如：描述埃菲爾鐵塔的基本結構和形狀。它有一個金屬框架，塔頂有一個尖塔，整個建築物呈現出一種網狀結構。
                它「看起來是什麼樣子」

        2. 風格圖像 (Style Image)
            想要生成的【擁有特定藝術風格的圖像】。
            例如：一幅埃菲爾鐵塔的畫作，但是這幅畫是用梵高的風格畫的。
            雖然依然能認出是埃菲爾鐵塔，但「風格完全不同」。



2. 內容損失 (Content Loss)

    內容損失是一種衡量【生成圖像】與【內容圖像】在特定卷積層上的【相似度】的度量方式。
    這種損失通常用於保持生成圖像的內容結構，使其與內容圖像接近。

    內容損失通常使用均方誤差(Mean Squared Error, MSE)計算：
        MSE = ∑( 真實值 - 預測值 ) / n


        
3. 風格損失 (Style Loss)

    風格損失是一種衡量【生成圖像】與【風格圖像】在風格層次上的【相似度】的度量方式。
    與內容損失不同，風格損失不關心具體的內容結構，而是關注圖像中的紋理、顏色分佈和藝術風格。

    1. Gram矩陣 (Gram Matrix)
        
        風格損失的計算依賴於Gram矩陣，這是一種表示【特徵之間相關性】的矩陣。
        Gram矩陣能夠捕捉到【特徵映射】之間的內部依賴關係，即一幅圖像的紋理信息。
        
        特徵映射 (Feature Map)：
            這是圖像經過CNN某層卷積運算後得到的輸出，它可以看作是圖像在該層的表示形式。
            圖像通過每一個卷積層時，該層會提取一些特定的特徵，這些特徵可以是邊緣、角點、紋理(保有空間訊息，即在圖片中的哪裏的特徵)。
            
            假設輸入圖像的大小為 ( H高度 x W寬度 x C通道 ) 通過卷積層。
                1. 首先得到能代表空間訊息的二維矩陣 (H' x W') 縮小的圖片，這些二維矩陣即為特徵圖 (Feature Maps)
                2. 多個二維特徵圖組合在一起形成了一個三維張量。這個三維張量的結構為(H' x W' x N)，N為卷積核(Kernel)數量，可以有不同的核，去抓出不同的特徵

                
    2. Gram矩陣的定義

        Gram矩陣是一個用來表示特徵映射之間相關性的矩陣。它通過計算【特徵映射的內積】來度量不同特徵之間的相似性。
        例如：特徵映射 F = N(特徵圖數量(深度)) x M(每個特徵圖空間大小(H' x W'))，Gram 矩陣 G 的計算公式為：

            G(i,j) = ∑ F(i,k) * F(j,k)  
                1. F(i,k): 特徵映射中第i個特徵圖在位置k上的值。
                2. F(j,k): 特徵映射中第j個特徵圖在位置k上的值。

                位置k: 特徵圖上用二維位置來描訴的像素點
            
            這樣計算出的Gram矩陣是一個N * N的矩陣，表示了不同特徵圖之間的相關性。


    3. 計算風格損失(Style Loss)

        風格損失是衡量【生成圖像的風格】與【目標風格圖像】之間的相似性的一種方式。
        為了計算風格損失，首先計算風格圖像和生成圖像在某些選定的卷積層上的Gram矩陣，然後比較這些Gram矩陣之間的差異。
        
        L(style) = {∑ [G(gen, ij) - G(style, ij)] ** 2} / 4(N**2)(M**2)
            1. G(gen, ij) 與 G(style, ij): 分別表示生成圖像和風格圖像的Gram矩陣。
            2. N 是特徵映射的通道數 (特徵圖的數量)。
            3. M 是特徵映射中每個特徵圖的空間大小。
'''
from __future__ import print_function  # 從 Python 2 向 Python 3 過渡的一個工具，確保 print 函數是 Python 3 的形式

import torch  
import torch.nn as nn  # 神經網路模塊
import torch.nn.functional as F  # 神經網路操作函數(如激活函數、損失函數)
import torch.optim as optim  

from PIL import Image  # 影像處理庫處理和操作影像。
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms  # 圖像轉換，用於對影像進行預處理，如縮放、裁剪等操作。
import torchvision.models as models  # 預訓練模型，用於加載和應用預訓練的神經網路模型。
import copy  # 用於進行深層複製。


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# 如果無 GPU 使用較小尺寸的圖像
imsize = 512 if torch.cuda.is_available() else 128  
# 根據設備選擇圖像大小。如果是 GPU，則使用 512x512 大小的圖像；如果是 CPU，則使用 128x128 大小的圖像，這樣可以減少計算負擔。

# 定義圖像預處理的轉換操作序列
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # 統一圖像尺寸為指定大小
    transforms.ToTensor()  # 將 PIL 圖像轉換為 tensor
])

# 定義一個函數，用於讀取圖像並將其轉換為 tensor
def image_loader(image_name):
    image = Image.open(image_name)  # 使用 PIL 打開圖像
    image = loader(image).unsqueeze(0)  # 使用預定義的轉換對圖像進行處理，然後使用 `unsqueeze(0)` 增加一個批次維度，因為 PyTorch 模型通常期望輸入有批次的維度
    return image.to(device, torch.float)  # 將圖像轉換為指定設備並設置為浮點數類型

# 定義一個反轉換操作，將 tensor 轉換回 PIL 圖像
unloader = transforms.ToPILImage()  

# 定義一個函數，用於顯示圖像
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # 先將張量從 GPU 移到 CPU，並創建一個副本，防止原始張量被修改
    image = image.squeeze(0)  # 去除批次維度，將形狀從 (1, C, H, W) 變為 (C, H, W)
    image = unloader(image)  # 使用反轉換操作將張量轉換為 PIL 圖像
    plt.axis('off')  # 隱藏坐標軸，讓圖片顯示得更清晰
    plt.imshow(image)  # 顯示圖像
    if title is not None:
        plt.title(title)  # 如果有提供標題，則設置圖像標題
    plt.pause(0.001)  # 暫停 0.001 秒，讓圖像顯示出來並及時更新


style_img = image_loader(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\StyleTransfer\des_glaneuses.jpg")  # 使用 image_loader 加載風格圖像，並將其轉換為 PyTorch 張量。
content_img = image_loader(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\StyleTransfer\dancing.jpg")  
print(style_img.shape, content_img.shape)  # 打印風格圖像和內容圖像的張量形狀，檢查它們的尺寸是否正確。
imshow(style_img, title='Style Image')  # 顯示風格圖像，並設置標題為 'Style Image'。
imshow(content_img, title='Content Image')  # 顯示內容圖像，並設置標題為 'Content Image'。

# 定義內容損失類，繼承自 nn.Module
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # 將目標內容從計算圖中分離出來
        # detach() 可以將張量從計算圖中分離，這樣在反向傳播時不會計算其梯度。
        # self.target 是一個固定的目標值，用於計算內容損失。
        self.target = target.detach()

    def forward(self, input):
        # 使用均方誤差損失(MSE)計算輸入和目標之間的差異
        # F.mse_loss 用於計算兩個張量之間的均方誤差
        self.loss = F.mse_loss(input, self.target)
        return input  # 返回輸入的張量，這允許這個模塊被插入到神經網絡中而不改變數據流


# 定義用於計算 Gram 矩陣的函數，這在風格轉移中用於捕捉風格特徵
def gram_matrix(input):
    # input.size()返回一個包含四個元素的元組，分別是 (batch_size, num_feature_maps, height, width)
    a, b, c, d = input.size()  # a 是批次大小(通常是 1)，b 是特徵圖的數量，c 和 d 是特徵圖的高度和寬度。

    # 將輸入的特徵圖重新排列，展平成二維張量，其中每一行代表一個特徵圖的展開
    features = input.view(a * b, c * d)  # view 用於重新調整張量的形狀，但不改變數據。

    # 計算特徵圖矩陣的 Gram 乘積，即兩個展開後的特徵圖矩陣之間的點積
    G = torch.mm(features, features.t())  # torch.mm 是 PyTorch 中的矩陣相乘操作，計算的結果是 Gram 矩陣。

    # 將 Gram 矩陣歸一化，這樣其值不會隨圖像尺寸或特徵圖數量而變動
    return G.div(a * b * c * d)  # 將計算出的 Gram 矩陣的每個元素除以總的元素數量，這樣可以防止隨著特徵圖數量的增加而增大。

# 定義風格損失類，繼承自 nn.Module
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # 計算目標特徵的 Gram 矩陣並將其從計算圖中分離出來
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        # 計算輸入圖像的 Gram 矩陣
        G = gram_matrix(input)
        # 計算風格損失，使用均方誤差(MSE)來比較輸入的 Gram 矩陣與目標 Gram 矩陣之間的差異
        self.loss = F.mse_loss(G, self.target)
        return input  # 返回輸入張量，允許損失模塊插入模型中不改變數據流

# 加載預訓練的 VGG19 模型，只保留卷積層部分，並將模型移動到指定的設備
cnn = models.vgg19(pretrained=True).features.to(device).eval()
# models.vgg19(pretrained=True) 會加載一個在 ImageNet 數據集上預訓練的 VGG19 模型。
# .features 提取出模型的卷積層部分，用於特徵提取。 


# 定義用於 CNN 的標準化均值和標準差
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)  # 均值基於 ImageNet 數據集計算的，用於對輸入圖像進行標準化。
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)  # 標準差基於 ImageNet 數據集，用於標準化。

# 定義標準化模塊，繼承自 nn.Module
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # 將均值和標準差轉換為合適的形狀，以便能與輸入的圖像張量直接相加或相減
        # .view(-1, 1, 1) 將張量調整為形狀 [C x 1 x 1]，-1是讓pytorch自動計算適當大小的意思， C 是通道數，這樣可適用於形狀為 [B x C x H x W] 的圖像張量
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # 對輸入圖像進行標準化處理：減去均值，除以標準差
        return (img - self.mean) / self.std  
    
# 指定用於計算內容損失的卷積層
content_layers_default = ['conv_4']  # conv_4 指在 VGG19 模型中用於計算內容損失的第四個卷積層
# 指定用於計算風格損失的卷積層
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # 這些是用於計算風格損失的卷積層列表，每一層都會對應計算一個風格損失


# 定義一個函數，用於構建風格轉移模型並設置內容和風格損失
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # 創建標準化模塊
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    # 初始化內容損失和風格損失的列表
    content_losses = []
    style_losses = []

    # 創建一個順序模型並首先加入標準化層
    model = nn.Sequential(normalization)

    i = 0  # 計數卷積層的數量
    for layer in cnn.children():  # 迭代 VGG19 模型中的所有層
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'  # 命名每一個卷積層
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            # 在這裡將 ReLU 層的 inplace 參數設置為 False，防止覆蓋原始輸入
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'  # 命名池化層
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'  # 命名批次正則化層
        else:
            # 如果層類型未被識別，拋出一個錯誤
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        # 將命名的層添加到模型中
        model.add_module(name, layer)

        # 如果該層是指定的內容層之一，則添加內容損失
        if name in content_layers:
            target = model(content_img).detach()  # 通過模型前向傳遞內容圖像，並將結果從計算圖中分離
            content_loss = ContentLoss(target)  # 創建內容損失模塊
            model.add_module(f"content_loss_{i}", content_loss)  # 將內容損失模塊添加到模型中
            content_losses.append(content_loss)  # 將內容損失添加到損失列表中

        # 如果該層是指定的風格層之一，則添加風格損失
        if name in style_layers:
            target_feature = model(style_img).detach()  # 通過模型前向傳遞風格圖像，並將結果從計算圖中分離
            style_loss = StyleLoss(target_feature)  # 創建風格損失模塊
            model.add_module(f"style_loss_{i}", style_loss)  # 將風格損失模塊添加到模型中
            style_losses.append(style_loss)  # 將風格損失添加到損失列表中

    # 在遍歷完所有層後，移除卷積層之後的任何層，只保留到最後一個損失層為止的模型部分
    for i in range(len(model) - 1, -1, -1):  # 從模型的最後一層向前遍歷
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break  # 找到最後一個損失層後停止遍歷

    model = model[:(i + 1)]  # 截斷模型，只保留到最後一個損失層為止的部分

    return model, style_losses, content_losses  # 返回模型，以及風格損失和內容損失的列表

# 定義一個函數，用於獲取優化器
def get_input_optimizer(input_img):
    # 為輸入圖像設置 LBFGS 優化器
    optimizer = optim.LBFGS([input_img])  # LBFGS 是一種適合於少量參數的優化算法，這裡用來優化生成圖像
    return optimizer  # 返回優化器


# 定義風格轉移的主函數
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    # 第一步：構建風格轉移模型，並初始化風格損失和內容損失
    print('Building the style transfer model..')
    
    # 調用 get_style_model_and_losses 函數來構建模型，並獲取風格損失和內容損失
    # cnn 是預訓練的 VGG19 模型，normalization_mean 和 normalization_std 是用來標準化圖像的均值和標準差
    # style_img 和 content_img 分別是風格圖像和內容圖像
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    # 第二步：準備輸入圖像進行優化，而不是優化模型參數
    input_img.requires_grad_(True)  # 設置輸入圖像張量的 requires_grad_ 為 True，允許對它進行梯度計算，以便可以通過反向傳播進行優化
    model.requires_grad_(False)  # 設置模型中的所有參數的 requires_grad 為 False，防止在優化過程中意外更新模型權重
    optimizer = get_input_optimizer(input_img)  # 使用 get_input_optimizer 函數來創建優化器，這裡優化的是輸入圖像而非模型參數

    # 第三步：開始優化過程
    print('優化 ..')
    run = [0]  # 初始化列表 run，用於記錄優化的步驟數，這裡使用列表是為了能夠在閉包函數中修改它

    # 使用 while 循環執行優化過程，直到達到指定的步驟數 num_steps
    while run[0] <= num_steps:
        # 定義一個閉包函數 closure，這是 LBFGS 優化器要求的，用於計算損失並反向傳播
        def closure():
            # 為了避免輸入圖像的像素值超出合法範圍，[0, 1] 表示正常的圖像像素值範圍
            with torch.no_grad():  # 禁用梯度計算
                input_img.clamp_(0, 1)  # 使用 clamp_ 函數將輸入圖像的所有像素值限制在 [0, 1] 之間

            # 重置優化器的梯度
            optimizer.zero_grad()  

            # 前向傳播，計算模型的輸出，這裡輸出不會被使用，但會觸發內容損失和風格損失的計算
            model(input_img)  
            
            # 初始化風格損失和內容損失
            style_score = 0  # 初始化風格損失
            content_score = 0  # 初始化內容損失

            # 累加所有風格損失
            for sl in style_losses:
                style_score += sl.loss  # 將每一層的風格損失累加到 style_score 中

            # 累加所有內容損失
            for cl in content_losses:
                content_score += cl.loss  # 將每一層的內容損失累加到 content_score 中

            # 根據預設的權重調整損失值
            style_score *= style_weight  # 根據指定的風格損失權重調整風格損失
            content_score *= content_weight  # 根據指定的內容損失權重調整內容損失

            # 計算總損失，這是風格損失和內容損失的加權和
            loss = style_score + content_score
            loss.backward()  # 對總損失進行反向傳播，計算出相對應的梯度，這些梯度將用來更新輸入圖像的像素值

            # 每隔 50 個步驟，打印當前的風格損失和內容損失，這有助於追蹤優化過程
            run[0] += 1  # 增加優化步驟的計數
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                # 打印當前的風格損失和內容損失
                # .item() 用於將包含單個元素的張量轉換為 Python 的數值類型(例如int、float)
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score  # 返回總損失，這是優化器所要求的

        optimizer.step(closure)  # 執行一次優化步驟，並使用 closure 函數來計算損失和梯度，這是 LBFGS 優化器的特性

    # 優化完成後，進行最後一次像素值範圍的限制
    with torch.no_grad():
        input_img.clamp_(0, 1)  # 再次將輸入圖像的像素值限制在 [0, 1] 的範圍內，確保生成的圖像是有效的

    return input_img  # 返回最終優化後的輸入圖像，這就是風格轉移的結果

# 使用內容圖像的副本作為輸入圖像，這樣可以保留原始內容圖像不變
input_img = content_img.clone()  # 使用 clone() 創建內容圖像的副本作為起始輸入圖像
# 執行風格轉移過程，生成最終的圖像
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

# 設置 Matplotlib 為交互模式，允許即時更新顯示的圖像
plt.ion()  # 打開交互式模式，使得 plt.show() 不會阻塞程序的執行

# 創建一個新的圖像窗口來顯示結果
plt.figure()
# 顯示最終生成的圖像，並設定圖像標題為 "Output Image"
imshow(output, title='Output Image')

plt.ioff()  # 關閉交互式模式，防止後續的 plt.show() 阻塞程序
plt.show()  # 顯示圖像，這是阻塞式的，直到窗口被關閉

# 加載新的風格圖像和內容圖像，用於進行下一次風格轉移實驗
style_img = image_loader(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\StyleTransfer\mirror.jpg")  
content_img = image_loader(r"C:\Users\user\Desktop\Python\PyTorch\Pytorch data\StyleTransfer\dancing.jpg") 
print(style_img.shape, content_img.shape)  # 風格圖像和內容圖像的尺寸，檢查加載是否成功
# 顯示新加載的風格圖像和內容圖像
imshow(style_img, title='Style Image')  # 顯示新的風格圖像，並設置標題為 "Style Image"
imshow(content_img, title='Content Image')  # 顯示新的內容圖像，並設置標題為 "Content Image"

# 再次使用內容圖像的副本作為輸入圖像，這樣我們可以保留原始內容圖像不變
input_img = content_img.clone()  # 使用 clone() 創建內容圖像的副本作為起始輸入圖像
# 執行風格轉移過程，生成最終的圖像
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

# 再次開啟 Matplotlib 的交互模式，允許即時更新顯示的圖像
plt.ion()  # 打開交互式模式，使得 plt.show() 不會阻塞程序的執行

# 創建一個新的圖像窗口來顯示結果
plt.figure()
# 顯示最終生成的圖像，並設定圖像標題為 "Output Image"
imshow(output, title='Output Image')

plt.ioff()  # 關閉交互式模式，防止後續的 plt.show() 阻塞程序
plt.show()  # 顯示圖像，這是阻塞式的，直到窗口被關閉



