'''
Selective Search Test
Selective Search是一種基於分層分割的目標檢測方法，用於生成一組潛在的候選區域。
這些候選區域中可能包含感興趣的目標，從而縮小後續目標檢測的搜索範圍。
Selective Search的核心思想是將圖像分割成若干較小的區域，然後逐步合併相鄰區域以生成不同尺度的候選區域。

1. 超像素分割
    超像素是圖像中的一個小區域，這個區域內的像素具有相似的顏色和紋理特徵。
    超像素分割將圖像分割成許多小區域，使每個區域內的像素更加均勻和一致。
    
    SLIC (Simple Linear Iterative Clustering) 是一種常用的超像素分割方法。
    它利用k-means聚類算法，根據顏色和空間距離將像素聚類成超像素。

    
2. 合併相鄰超像素
    在生成超像素後，需要提取每個超像素的特徵，這些特徵用於評估超像素之間的相似性。常見的特徵包括：

    1. 顏色特徵：通常使用顏色直方圖來表示超像素的顏色分布。
    2. 紋理特徵：使用灰度共生矩陣 (GLCM) 等方法來提取紋理特徵。

    
3. 合併相鄰區域
    根據提取的特徵，逐步合併相鄰的超像素，這個過程稱為區域合併。區域合併基於以下幾種相似性度量：

    1. 顏色相似性：計算相鄰超像素顏色直方圖的相似性。
    2. 紋理相似性：計算相鄰超像素紋理特徵的相似性。
    3. 大小相似性：較小的區域更容易被合併。
    4. 填充度：合併後區域的邊界應該盡量緊湊。
    
    合併過程是逐步的，每次合併最相似的一對相鄰區域，直到合併達到一定的停止條件。

    
4. 生成候選區域
    通過逐步合併相鄰的超像素，我們可以生成多個不同尺度的候選區域。這
    些候選區域可以覆蓋圖像中的所有潛在目標，從而提供一組有潛力的區域供後續的目標檢測模型進行精細檢測。
'''

import cv2  
import sys  # 訪問命令行參數

# 讀取影像
img_path = r'PyTorch/Pytorch data/images_Object_Detection/bike2.jpg'  
if len(sys.argv) > 1:  # 如果命令行參數包含影像路徑
    img_path = sys.argv[1]  # 使用命令行參數中的影像路徑
img = cv2.imread(img_path)  # 讀取影像

# 執行 Selective Search
cv2.setUseOptimized(True)  # 啟用OpenCV優化
cv2.setNumThreads(8)  # 設置OpenCV的線程數量為8
gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # 創建Selective Search分割對象
gs.setBaseImage(img)  # 設置基礎影像

select_mode = 'f'  # 設置默認選擇模式為'快速'
if len(sys.argv) > 2 and sys.argv[2] == 's':  # 如果命令行參數包含選擇模式
    gs.switchToSingleStrategy()  # 切換到單策略模式
elif len(sys.argv) > 2 and sys.argv[2] == 'q':  # 如果選擇模式為'高質量'
    gs.switchToSelectiveSearchQuality()  # 切換到高質量模式
else:  # 否則
    gs.switchToSelectiveSearchFast()  # 切換到快速模式

rects = gs.process()  # 執行Selective Search並返回候選區域
print('個數:', len(rects))  # 打印候選區域的數量
nb_rects = 10  # 設置初始顯示的候選區域數量

# 畫框
while True:  # 進入無限循環
    wimg = img.copy()  # 複製影像

    for i in range(len(rects)):  # 遍歷所有候選區域
        if i < nb_rects:  # 如果候選區域數量小於設定值
            x, y, w, h = rects[i]  # 獲取候選區域的座標和尺寸
            cv2.rectangle(wimg, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)  # 在影像上畫矩形框

    cv2.imshow("Output", wimg)  # 顯示影像
    key = cv2.waitKey()  # 等待按鍵輸入

    if key == 43:  # 如果按下'+'
        nb_rects += 10  # 增加顯示的候選區域數量

    elif key == 45 and nb_rects > 10:  # 如果按下'-'且顯示的候選區域數量大於10
        nb_rects -= 10  # 減少顯示的候選區域數量

    elif key == 113:  # 如果按下'q'
        break  # 跳出循環

cv2.destroyAllWindows()  # 關閉所有OpenCV視窗
