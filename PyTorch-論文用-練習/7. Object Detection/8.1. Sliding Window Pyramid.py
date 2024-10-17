'''
在大多數圖像處理系統(包括 OpenCV)中，圖像的座標系統：

原點(0, 0)位於圖像的左上角。
x 軸從左到右遞增。
y 軸從上到下遞增。
'''

'''
yield 是 Python 的生成器(generator)功能，
它允許函數在每次調用 yield 時返回一個值，而函數的狀態會被保留，
下次調用時會從這個狀態繼續執行，而不是重新開始函數。
'''
#!/usr/bin/env python
# shebang，用於指示系統應該使用哪個解釋器來執行這個腳本

# coding: utf-8

# # 範例1. 對圖片滑動視窗並作影像金字塔
# ### 原程式來自Sliding Windows for Object Detection with Python and OpenCV (https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)

# 載入套件
import cv2 # OpenCV 的 Python 介面，用於圖像處理。
import time # 用於暫停程式執行
import imutils  # 提供一些便捷的影像處理函數，特別是用於 OpenCV

# 影像金字塔操作
# image：原圖
# scale：每次縮小倍數
# minSize：最小尺寸
def pyramid(image, scale=1.5, minSize=(30, 30)):
    # 第一次傳回原圖
    yield image # 傳回原圖
    # yield 是一種生成器的用法，它可以記住函數的狀態並在下一次調用時繼續執行

    while True: 
        # 計算縮小後的圖像寬度，image.shape[1] 是原圖的寬度
        w = int(image.shape[1] / scale)
        # 使用 imutils.resize 函數將圖像按比例縮小到寬度 w, 高度會按比例自動調整
        image = imutils.resize(image, width=w)
        # 檢查縮小後的圖像是否達到最小尺寸，如果高度或寬度小於 minSize，則break
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # 傳回縮小後的圖像
        yield image

# 滑動視窗 
# image: 輸入的圖像, stepSize: 滑動的步長, windowSize: 視窗的大小
def sliding_window(image, stepSize, windowSize):    
    for y in range(0, image.shape[0], stepSize):     # 在 y 軸方向每次滑動 stepSize 個像素
        for x in range(0, image.shape[1], stepSize): # 在 x 軸方向每次滑動 stepSize 個像素
            # 傳回裁剪後的視窗
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]]) # 生成當前滑動視窗的左上角座標 (x, y) 以及視窗內的圖像內容 image[y:y + windowSize[1], x:x + windowSize[0]]
            # y: 滑動視窗的頂部邊界
            # y + windowSize[1]: 滑動視窗的底部邊界, windowSize[1] 是滑動視窗的高度
            # 表示從 y 開始，到 y + windowSize[1] 結束，這範圍是視窗的高度

            # x: 滑動視窗的左邊界
            # x + windowSize[0]: 滑動視窗的右邊界，windowSize[0] 是滑動視窗的寬度
            # 表示從 x 開始，到 x + windowSize[0] 結束，這範圍是視窗的寬度

# ## 測試
# 讀取一個圖檔
image = cv2.imread(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_Object_Detection\lena.jpg')

# 設定滑動視窗的寬度和高度
(winW, winH) = (128, 128)

# 取得影像金字塔各種尺寸
for resized in pyramid(image, scale=1.5): # pyramid 函數生成一系列不同尺寸的圖像, resized 變數依次取得這些圖像
    # 滑動視窗
    for (x, y, window) in sliding_window(resized, stepSize=32, 
                                         windowSize=(winW, winH)):
        # sliding_window 函數生成一系列滑動視窗，返回每個視窗的左上角座標 (x, y) 和視窗內容 window

        # 視窗尺寸不合即放棄，滑動至邊緣時，尺寸過小
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # 標示滑動的視窗
        clone = resized.copy() # 創建 resized 圖像的副本
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2) # cv2.rectangle 函數用於在圖像上畫一個矩形
        # 左上角 (0,0) 是圖像的原點
        # clone：要畫矩形的圖像
        # (x, y)：矩形的左上角座標
        # (x + winW, y + winH)：矩形的右下角座標
        # (0, 255, 0)：矩形的顏色, 表綠色
        # 2：矩形的線條寬度(以像素為單位)
        
        cv2.imshow("Window", clone) # 在名為 Window 的窗口中顯示圖像 clone
        
        cv2.waitKey(1) # 程序會等待 1 毫秒以捕獲鍵盤事件, 主要作用顯示滑動的窗口有時間刷新, 如果按一下的話會直接關掉
        # 暫停
        time.sleep(0.025) # 程序暫停 25 毫秒

# 結束時關閉視窗        
cv2.destroyAllWindows()