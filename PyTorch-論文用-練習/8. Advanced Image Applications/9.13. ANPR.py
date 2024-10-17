'''
1. 車牌自動辨識 (Automatic Number Plate Recognition, ANPR)
    識別和讀取車輛車牌號碼，流程包括圖像捕獲、車牌檢測、車牌預處理、字符識別以及結果輸出。

    1. 圖像捕獲與輸入
        從輸入中捕獲包含車輛的圖像。

        
    2. 車牌檢測
        在整個圖像中定位車牌的位置。
        將整個圖像縮小到只包含車牌的部分，這樣可以提高後續字符識別的準確性。

        1. 邊緣檢測：
            車牌通常是矩形且包含字符，這使得車牌在圖像中與背景相比有明顯的邊緣特徵。
            通過邊緣檢測技術，系統可以識別出圖像中的邊緣並區分出可能是車牌的區域。

        2. 輪廓檢測：
            在邊緣檢測後，輪廓檢測技術可以用來確定邊緣區域的形狀和大小，從而識別出車牌的矩形區域。

        3. 幾何過濾：
            車牌的形狀和比例是相對固定的，因此通過幾何過濾可以進一步排除不符合車牌特徵的區域。(例如排除過於小或過於大的矩形區域。)

            
    3. 車牌預處理
        在車牌檢測後，獲得了包含車牌的子圖像。
        為了提高字符識別的效果，需要對車牌圖像進行進一步的預處理。

        1. 灰度化：
            將彩色圖像轉換為灰度圖像，這樣可以簡化計算並突出字符的對比度。

        2. 二值化：
            將灰度圖像轉換為只有黑白兩色的二值圖像。
            二值化可以更好地分離字符(通常為黑色)和背景(通常為白色。)

        3. 去噪：
            在圖像處理過程中，噪點會影響字符識別的準確性。需要去除這些干擾信息。


    4. 字符識別(ORC)
        識別車牌上的字符

    
    5. 結果輸出

'''
'''
import cv2  
import imutils  # imutils 庫，用於簡化 OpenCV 操作的工具函數
import numpy as np  
import matplotlib.pyplot as plt 
import pytesseract # 用於 OCR 
from PIL import Image  # 用於圖像處理

# 載入圖檔
image = cv2.imread(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_ocr\2.jpg', cv2.IMREAD_COLOR)  
# cv2.imread() 讀取圖像，cv2.IMREAD_COLOR 表示以彩色模式讀取圖像，讀取後的圖像會以 BGR 色彩格式儲存

# 顯示圖檔
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR 轉 RGB 
plt.imshow(image_RGB)  
plt.axis('off')  
plt.show()  



# 車牌號碼 OCR 辨識
char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'  
# 設定白名單字符 (char_whitelist)，只允許辨識車牌號碼中可能出現的字母和數字

text = pytesseract.image_to_string(image, config=
           f'-c tessedit_char_whitelist={char_whitelist} --psm 6 ')  
# pytesseract.image_to_string() 進行 OCR 辨識，將圖像轉換為文字
# config 參數用於傳遞 Tesseract 的設定
# --psm 6 表示 Tesseract 將圖像視為一個均勻排列的區塊，適合單行文字的辨識
print("車牌號碼：", text) # print 辨識出的車牌號碼



# 萃取輪廓
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
# cv2.cvtColor() 將圖像轉換為灰階，輪廓萃取的前處理步驟

gray = cv2.bilateralFilter(gray, 11, 17, 17)  
# cv2.bilateralFilter() 進行雙邊濾波，模糊圖像以減少雜訊，同時保持邊緣清晰

edged = cv2.Canny(gray, 30, 200)  
# cv2.Canny() 進行邊緣檢測，萃取圖像中的邊緣輪廓
# 30 和 200 是 Canny 邊緣檢測的閾值


# 顯示圖檔
plt.imshow(edged, cmap='gray') # plt.imshow() 顯示檢測出的邊緣圖像，並設置 cmap='gray' 以灰階模式顯示
plt.axis('off')  
plt.show()  


# 取得等高線區域，並排序，取前10個區域
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
# cv2.findContours() 在邊緣圖像中找到所有等高線，返回值為等高線和階層
# cv2.RETR_TREE 表示提取所有輪廓並構建完整的階層
# cv2.CHAIN_APPROX_SIMPLE 表示壓縮水平、垂直和對角線上的輪廓點，只保留端點

cnts = imutils.grab_contours(cnts)  
# imutils.grab_contours() 來兼容不同版本 OpenCV 的等高線返回格式

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  
# sorted() 根據輪廓的面積 (cv2.contourArea) 將等高線從大到小排序，並取出前10個最大的輪廓

cnts[0].shape  
# 檢查最大的輪廓的形狀 (即輪廓點的數量)，這裡的 shape 會返回一個元組，表示輪廓中包含的點數

# 找第一個含四個點的等高線區域
screenCnt = None  # 初始化變數 screenCnt，用於儲存找到的四邊形輪廓
for i, c in enumerate(cnts):  
    # enumerate() 迴圈遍歷排序後的輪廓列表，i 是索引，c 是當前輪廓
    # 計算等高線區域周長
    peri = cv2.arcLength(c, True)  
    # cv2.arcLength() 計算輪廓的周長，第二個參數 True 表示輪廓是封閉的
    # 轉為近似多邊形
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)  
    # cv2.approxPolyDP() 將輪廓近似為多邊形，0.018 * peri 表示近似精度，數值越小近似越精確
    # True 表示輪廓是封閉的
    # 等高線區域維度
    print(c.shape)  
    # 輸出當前輪廓的形狀，即輪廓中點的數量

    # 找第一個含四個點的多邊形
    if len(approx) == 4:  
        # 如果近似多邊形有四個頂點，表示找到了一個四邊形，通常是車牌區域
        screenCnt = approx  # 將這個四邊形輪廓賦值給 screenCnt
        print(i)  # 輸出找到的四邊形在輪廓列表中的索引
        break  # 找到第一個符合條件的輪廓後立即跳出迴圈


# 在原圖上繪製多邊形，框住車牌
if screenCnt is None:  # 檢查是否找到符合條件的四邊形輪廓
    detected = 0  # 如果沒有找到，設置 detected 為 0，表示未偵測到車牌
    print("No contour detected")  # 輸出提示信息，未偵測到輪廓
else:
    detected = 1  # 如果找到四邊形輪廓，設置 detected 為 1，表示已偵測到車牌
    
if detected == 1:  # 如果偵測到車牌
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)  
    # cv2.drawContours() 函數在原圖上繪製輪廓，使用綠色 (0, 255, 0) 並設置線條寬度為 3
    # -1 表示繪製整個輪廓
    print(f'車牌座標=\n{screenCnt}')  

# 去除車牌以外的圖像
mask = np.zeros(gray.shape, np.uint8)  
# 創建一個與灰階圖像相同大小的全黑遮罩圖像，使用 np.zeros() 將其像素值全設為 0
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)  
# 在遮罩圖像上繪製車牌的輪廓，並將車牌區域填充為白色 (255)
new_image = cv2.bitwise_and(image, image, mask=mask)  
# cv2.bitwise_and() 將原圖與遮罩進行按位與操作，只保留車牌區域，其他區域變為黑色

# 轉為浮點數
src_pts = np.array(screenCnt, dtype=np.float32)  
# 將車牌的四邊形頂點座標轉換為浮點數格式，這是仿射變換所需的格式

# 找出車牌的上下左右的座標
left = min([x[0][0] for x in src_pts])  # 計算車牌最左邊的 x 座標
right = max([x[0][0] for x in src_pts])  # 最右邊的 x 座標
top = min([x[0][1] for x in src_pts])  # 最上面的 y 座標
bottom = max([x[0][1] for x in src_pts])  # 最下面的 y 座標

# 計算車牌寬高
width = right - left  # 計算車牌的寬度
height = bottom - top  # 車牌的高度
print(f'寬度={width}, 高度={height}')  # 輸出車牌的寬和高

# 計算仿射(affine transformation)的目標區域座標，須與擷取的等高線區域座標順序相同
if src_pts[0][0][0] > src_pts[1][0][0] and src_pts[0][0][1] < src_pts[3][0][1]:
    # 如果 src_pts 的第一個頂點 x 座標大於第二個頂點，且 y 座標小於第四個頂點
    print('起始點為右上角')  # 起始點為右上角
    dst_pts = np.array([[width, 0], [0, 0], [0, height], [width, height]], dtype=np.float32)  
    # 設置目標區域的四個頂點，與原圖中車牌頂點的順序一致
elif src_pts[0][0][0] < src_pts[1][0][0] and src_pts[0][0][1] > src_pts[3][0][1]:
    # 判斷條件：如果 src_pts 的第一個頂點 x 座標小於第二個頂點，且 y 座標大於第四個頂點
    print('起始點為左下角')  # 起始點為左下角
    dst_pts = np.array([[0, height], [width, height], [width, 0], [0, 0]], dtype=np.float32)  
    # 設置目標區域的四個頂點，與原圖中車牌頂點的順序一致
else:
    print('起始點為左上角')  # 默認情況下起始點為左上角
    dst_pts = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)  
    # 設置目標區域的四個頂點，與原圖中車牌頂點的順序一致
    
# 仿射變換
M = cv2.getPerspectiveTransform(src_pts, dst_pts)  
# cv2.getPerspectiveTransform() 計算仿射變換矩陣 M，將原始車牌區域變換為目標矩形區域
Cropped = cv2.warpPerspective(gray, M, (int(width), int(height)))  
# cv2.warpPerspective() 將圖像進行仿射變換，生成一個正方形的車牌圖像

# 車牌號碼 OCR 辨識
char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'  
# 設定白名單字符 (char_whitelist)，只允許辨識車牌號碼中可能出現的字母和數字

text = pytesseract.image_to_string(Cropped, config=
           f'-c tessedit_char_whitelist={char_whitelist} --psm 6 ')  
# 使用 pytesseract.image_to_string() 進行 OCR 辨識，將仿射變換後的車牌圖像轉換為文字
# config 參數用於傳遞 Tesseract 的設定 (psm 6)
# --psm 6 表示 Tesseract 將圖像視為一個均勻排列的區塊，適合單行文字的辨識
print("車牌號碼：", text)  # print 辨識出的車牌號碼

# 顯示原圖及車牌
cv2.imshow('Orignal image', image)  # cv2.imshow() 顯示原始圖像，標註了車牌位置
cv2.imshow('Cropped image', Cropped)  # cv2.imshow() 顯示仿射變換後的車牌圖像

# 車牌存檔
cv2.imwrite('Cropped.jpg', Cropped)  # cv2.imwrite() 將截取的車牌圖像保存為 'Cropped.jpg'

cv2.waitKey(0)  # cv2.waitKey(0) 等待使用者按下任意鍵，# 按 Enter 鍵結束

# 關閉所有視窗
cv2.destroyAllWindows()  # 關閉所有由 OpenCV 開啟的視窗

'''