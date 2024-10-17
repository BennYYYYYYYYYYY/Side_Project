'''
一般深度學習的model安裝選項:
 (1) 本地伺服器(Local Server)
 (2) 雲端伺服器(Cloud Server)
 (3) 邊緣運算(IoT Hub)
'''

# 以下就網頁開發建立一個手寫阿拉伯數字辨識網站
# 安裝 Streamlit套件
# pip install streamlit

# 執行此Python程式, 必須在終端 以streamlit run開頭, 而非python執行
# streamlit run '檔案路徑'


# 網頁顯示後, 拖曳myDigits目錄內的任一檔案至畫面中的上傳圖檔區域, 就會顯示辨識結果, 也可以使用繪圖軟件書寫數字

import streamlit as st
from skimage import io # 主要用於圖像的輸入與輸出
from skimage.transform import resize 
import numpy as np
import torch

# 模型載入: 其中@st.cache可以將模型儲存至快取(Cache), 避免每次請求都至硬碟讀取, 浪費時間
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@st.cache(allow_output_mutation=True) # 將模型儲存至Cache, 意味著當相同的函數被調用時, streamlit不會重新執行函數, 而是直接使用緩存結果
def load_model(): # 定義函數是下載訓練好的model
    return torch.load('C:\\Users\\user\\Desktop\\Python\\PyTorch\\CH4\\model.pt').to(device)

model = load_model() # 下載已訓練model

# 標題
st.title("上傳圖片(0~9)辨識") # 新增標題

# 上傳圖片
upload_file = st.file_uploader('上傳圖片(.png)', type='png') 
# st.file_uploader: 創建一個文件上傳控件。使用者可以通過這個控件上傳文件，並且該文件將被此函數返回，以便進一步處理或分析。
# '上傳圖片(.png)': 顯示給用戶看的標題資訊
# type='png': 指定可接受的上傳文件類型
'''
上傳控件（File Uploader Widget）是一種用戶介面元件，允許用戶從他們的本地電腦選擇文件並上傳到網站或應用程式中。
在網頁開發和應用程式設計中，這種控件提供了一個直觀的界面，讓用戶能夠輕鬆地上傳文件，例如文檔、圖片、視頻等。
'''

# 檔案上傳後, 執行下列工作
if upload_file is not None: # 如果檔案有被上傳:
    image1 = io.imread(upload_file, as_gray=True) # 基本作用是將圖像文件加載到内存中，以便進一步處理或分析。
    # io.imread函式: 從上傳的檔案中讀取圖像文件, 並將其轉換為灰階

    # 縮為(28, 28)大小的影像
    image_resize = resize(image1, (28,28), anti_aliasing=True) # 縮放成28x28像素, 且縮放過程開啟抗鋸齒
    x1 = image_resize.reshape(1,28,28) # 多加一維(batch_size)

    # 反轉顏色
    x1 = torch.FloatTensor(1-x1).to(device) # 轉為FloatTensor, 並反轉顏色

    # 預測
    prediction = torch.softmax(model(x1), dim=1) # x1丟入模型後輸出機率(2維)

    # 顯示預測結果
    st.write(f'### 預測結果:{np.argmax(prediction.detach().cpu().numpy())}') # 預測結果是機率最大的那個
    # st.write(): 將結果顯示在應用介面上

    # 顯示上傳圖檔
    st.image(image1) # 顯示原本上傳的圖片
