'''
1. 光學影像辨識 (Optical Character Recognition, OCR)
    用於將印刷或手寫的文本從圖像中提取出來，並轉換為可以機器讀取和處理的數字文本。
    這項技術應用廣泛，如掃描文件、識別車牌、處理票據、翻譯書籍等。


2. Tesseract OCR
    Tesseract是一款開源的OCR引擎，最初由 HP Labs開發，後來轉由Google維護和改進。
    Tesseract支持多種語言和字體，能夠從圖像中提取出文本，並將其轉換為可編輯的文本格式。


3. Tesseract OCR的工作流程

    1. 圖像輸入 (Image Input)
        Tesseract接收一個包含文本的圖像文件作為輸入(可以是掃描的文件、照片、截圖等)。

        
    2. 圖像預處理 (Image Preprocessing)
        進行光學影像辨識之前，圖像需要進行一些預處理，以提高識別的準確性。
            
            1. 灰度化 (Grayscale Conversion)
                將圖像轉換為灰度圖像，減少計算量。

            2. 二值化 (Binarization)
                將灰度圖像轉換為只有黑白兩色的圖像。
                用的二值化方法是Otsu’s Method，它可以自動選擇閾值將圖像分割為前景(文本)和背景。

            3. 去噪 (Noise Removal)
                移除圖像中的噪點，以避免誤認為字符的一部分。

                
    3. 文字區域檢測 (Text Detection)
        通過分析像素的排列來識別出文本塊、行和單詞。

            1. 連通組件分析 (Connected Component Analysis)
    
            
    4. 字符分割 (Character Segmentation)
        檢測到文本區域後，Tesseract將這些區域進一步細分為單個字符。
        字符分割的結果對於後續的識別準確性至關重要。


    5. 特徵提取 (Feature Extraction)
        Tesseract對每個字符進行特徵提取。
        特徵提取的目的是將字符轉換為一組可以用於識別的數據特徵。

        1. 輪廓、線條和區域特徵：
            Tesseract會提取字符的幾何特徵，如邊界、輪廓、筆劃方向等，這些特徵將用於模式匹配過程中。

    
    6. 字符識別 (Character Recognition)
        這是Tesseract的核心部分，通過與預先訓練的字符模型進行比對，將提取的特徵轉換為對應的字符。

        1. 判斷性識別 (Adaptive Recognition)
            Tesseract首先使用一個預訓練的模型來識別字符，這個模型包含了大量不同語言和字體的字符數據。

        2. 自適應識別 (Adaptive Classifier)
            在初步識別後，Tesseract會根據輸入圖像的特定字體和特徵進行自適應學習，進一步提高識別的準確性。


    7. 後處理 (Post-processing)
        Tesseract對結果進行後處理，以修正識別過程中的潛在錯誤。
        (例如利用字典檢查來校正可能的拼寫錯誤，或者根據上下文推斷不確定的字符。)

    
    8. 輸出 (Output)
        將識別結果輸出為文本文件、PDF文件或其他格式。
'''
'''
# 載入相關套件
import cv2  # 用來處理影像的相關功能
import pytesseract  # 用來進行 OCR，它是 Tesseract OCR 引擎的 Python 包裝
import matplotlib.pyplot as plt  

# 載入圖檔
image = cv2.imread(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_ocr\receipt.png')  
# cv2.imread() 讀取指定路徑的圖像檔案，將其儲存在變數 image 中

# 顯示圖檔
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
# cv2.cvtColor() 從 BGR 轉為 RGB 

plt.figure(figsize=(10,6))  
# plt.figure() 設置圖像的視窗大小，設定為寬 10 、高 6 

plt.imshow(image_RGB)  
# plt.imshow() 顯示轉換後的 RGB 圖像

plt.axis('off')  
plt.show()  



# 參數設定
custom_config = r'--psm 6'  
# 設定 OCR 的參數，這裡使用的是 '--psm 6'，表示 Tesseract 假設圖像是一個統一的塊，包含可變行間距的文本。換句話說，這種模式適合用於文本塊的處理，特別是在文本可能包含多行且行距不一致的情況下。
# --psm 是 Page Segmentation Mode 的簡寫，可以控制 Tesseract 如何解析圖像中的文字

# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config)) # print 出識別結果
# pytesseract.image_to_string() 進行 OCR，將圖像中的文字識別出來並轉換為字串
# 傳入的參數包括圖像 image 和自訂的配置 custom_config
'''
'''
Tesseract PSM (Page Segmentation Mode, 頁面分割模式) 模式

    0: 僅進行頁面的方向和文字編排的檢測，但不執行 OCR 操作。
        只想知道頁面上的文本方向或編排風格，而不需要識別文字內容時使用。

    1: 自動執行頁面分割並進行方向檢測，同時處理 OCR。
        適合結構複雜的文件，Tesseract 自動決定如何分割和處理整個頁面，包括方向和文字編排。

    2: 僅進行頁面分割，不進行方向檢測或 OCR。
        讓 Tesseract 分析頁面佈局，但不進行實際的文本識別時使用。

    3: 完全自動頁面分割，但不進行方向檢測，進行 OCR 處理。
        標準的文件頁面，且頁面方向已知，適合常見的文檔格式。

    4: 假設圖像包含一個單一的文本列，行大小可能變化。
        適合處理書本或報紙的單欄文本，行距可能不同。

    5: 假設圖像是一個單一的文本塊，且行距一致。
        適合處理以一致間距排列的文本，如打印的書籍或報紙段落。

    6: 假設圖像是一個統一的文本塊，但行距可能不一致。
        適合處理可能具有變動行距的多行文本塊，如手寫文本或多行打印文本。

    7: 假設圖像是單行文本。
        適合處理僅有一行文本的圖像，如標籤或橫幅。

    8: 假設圖像是一個單詞。
        適合處理非常簡單的圖像，僅有一個單詞的情況。

    9: 假設圖像是一個圓形的單詞，通常是印章或章印中的文字。
        適合識別圓形佈局的單詞，如公章或印章中的文字。

    10: 假設圖像僅包含一個字符
        適合處理單字符的圖像，如手寫字母或單字識別。

    11: 在圖像中儘可能找到所有文本，無需特別的順序。
        適合處理文本零散且無特定排列的圖像，如塗鴉或散佈在圖像各處的文本。

    12: 在執行第11模式的基礎上，還進行方向檢測。
        適合處理文本零散且無特定排列的圖像，並且需要檢測文字方向。

'''

'''
# 參數設定，只辨識數字
custom_config = r'--psm 6 outputbase digits'  
# 新的 OCR 參數，這次加上了 'outputbase digits'，表示 Tesseract 只辨識數字

print(pytesseract.image_to_string(image, config=custom_config))  
# 使用與上面相同的方法進行 OCR 辨識，但這次的配置只會提取數字




# 參數設定白名單，只辨識有限字元
custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6'  
# 設定另一組 OCR 參數，這次使用 '-c tessedit_char_whitelist=...'，來設置一個白名單
# 只允許辨識特定的字母 (現為小寫英文字母)

print(pytesseract.image_to_string(image, config=custom_config))  
# 再次使用 pytesseract.image_to_string() 進行 OCR 辨識，這次只會辨識白名單中的字元



# 參數設定黑名單，只辨識有限字元
custom_config = r'-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz --psm 6'  
# 設定 OCR 的參數，這次使用了 '-c tessedit_char_blacklist=...'，來設置一個黑名單
# Tesseract 將不會辨識黑名單中的字元(小寫英文字母)，其餘字元會正常識別

# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))  
# 使用 pytesseract.image_to_string() 進行 OCR（光學字元辨識），
# 這次不會辨識黑名單中的字元，並使用 print() 輸出識別結果





# 載入圖檔
image = cv2.imread(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_ocr\chinese.png')  

# 顯示圖檔
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
# cv2.cvtColor() BGR 轉 RGB 

plt.figure(figsize=(10,6))  
# 使用 plt.figure() 設置顯示圖像的視窗大小，這裡設定為寬 10 、高 6 

plt.imshow(image_RGB)  
plt.axis('off')  
plt.show()  



# 辨識多國文字，中文繁體、日文及英文
custom_config = r'-l chi_tra+jpn+eng --psm 6'  
# 設定 OCR 的參數，這裡使用了 '-l' 參數來指定語言包
# 'chi_tra' 是繁體中文，'jpn' 是日文，'eng' 是英文
# 通過組合這三個語言包，Tesseract 可以同時識別這三種語言的文字

# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))  
# pytesseract.image_to_string() 進行 OCR
# 這次會識別圖像中的繁體中文、日文和英文



# 載入圖檔
image = cv2.imread(r'C:\Users\user\Desktop\Python\PyTorch\Pytorch data\images_ocr\chinese_2.png')  

# 顯示圖檔
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
plt.figure(figsize=(10,6))  
plt.imshow(image_RGB)  
plt.axis('off')  
plt.show()  




# 辨識多國文字，中文繁體、日文及英文
custom_config = r'-l chi_tra+jpn+eng --psm 6'  
# 使用相同的 OCR 參數來指定語言包，辨識繁體中文、日文和英文

# OCR 辨識
print(pytesseract.image_to_string(image, config=custom_config))  


'''




