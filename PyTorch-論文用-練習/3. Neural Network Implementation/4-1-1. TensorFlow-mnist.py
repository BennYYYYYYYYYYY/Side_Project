'''
1. 讀取手寫阿拉伯數字的影像，影像中的每一個像素當成一個特徵，每筆資料為寬高(28,28)的點陣圖形
2. 建立神經網路模型，利用梯度下降法(Gradient Descent)求解模型參數，一般稱為權重(Weight)
3. 依照模型去推斷每一個影像是0~9的機率，再以最大機率者為預測的結果
'''
# 首先使用TensorFlow的手寫阿拉伯數字辨識
# pip install tensorflow
import tensorflow as tf
mnist = tf.keras.datasets.mnist # 從TensorFlow的 Keras API 中獲取"MNIST"數據集

# 匯入MNIST手寫 訓練資料
(x_train, y_train),(x_test, y_test) = mnist.load_data() #載入MNIST數據集並分成訓練集和測試集
# mnist.load_data()：這個函數負責加載 MNIST 數據集
# 它返回兩個元組：一個是訓練集 (x_train, y_train)，另一個是測試集 (x_test, y_test)

''' 
圖像數據的標準化(歸一化)

圖像數據通常以整數儲存，每個像素值從0~255，代表不同的灰度級別(RGB彩色通道)
歸一化是將其轉換為0~1的浮點數
'''
# 特徵縮放至(0, 1)之間, 除以255使其在0~1之間
x_train, x_test = x_train/255.0, x_test/255.0  


# 建立模型
model = tf.keras.models.Sequential([    # 建立一個順序模型
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # 把(28, 28)的資料flatten成 一維784個數值的向量(28*28)
    # 將圖像從二維的形式（28x28像素）轉換成一維的形式
    tf.keras.layers.Dense(128, activation='relu'),  #  建立全連結層(Dense layer)： 層中的每個神經元都與前一層中的每個神經元相連接
    # 在此案例中，層有128個神經元。此層會學習128個特徵，最後輸出128個值
    # Activation Function(激活函數)：  引入非線性性質到模型中
    
    # 在此案例中使用"ReLU函數"（Rectified Linear Unit）: 數學表示為f(x)=max(0,x)
    # (1) 當輸入 x>0 時,輸出為x
    # (2) 當輸入 x<=0 時, 輸出為0
    tf.keras.layers.Dropout(0.2), # Dropout正則化: 在此被設定為20%, 即每個神經元有20%概率被隨機選擇並設置輸出為0
    # Dropout: 一種防止神經網路overfitting的正則化技術
    # Dropout通過在訓練過程中隨機"丟棄"（輸出為0）一部分神經元的方式
    # 減少模型對訓練數據中任何單一樣本的依賴，從而增加模型的泛化能力。
    tf.keras.layers.Dense(10, activation='softmax') 
    # 全連接層: 具有10個神經元, 對應於10個類別的概率 (0~9)
    # 激活函數使用softmax將輸出轉換為機率分布，所有總類的sum=1(100%)
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(matrix)
model.compile(optimizer='adam', # 優化器設定adam梯度下降優化算法
              loss='sparse_categorical_crossentropy', 
              # 基於交叉熵損失（crossentropy loss）的一種。交叉熵損失是一種衡量模型預測概率分布與真實標籤概率分布之間差異的方法。
              # 對於多類別分類問題，交叉熵損失特別有效，因為它能夠衡量模型對於每個類別預測的準確度。

              # sparse: 反映了類別標籤的表示方式
               # (1) 獨熱編碼 (One-hot encoding): 每個類別由一個等長於類別數量的向量表示，其中一個元素為1，其餘為0
               # (2) 整數標籤 (Sparse labels): 每個類別直接用一個整數表示，1,2,3,4,5,6,...
              metrics=['accuracy']) # 評估指標(metrics)
# model.compile(): 配置模型的學習過程

# 模型訓練, epochs: 訓練週期, validation_split: 驗證資料佔20%
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
# 透過 model.fit 可以將訓練數據提供給model, 並指定跌代次數, 和驗證集比例
 
# (1) x_train: 訓練用的(28, 28)手寫數據
# (2) y_train: x_train相對應的數字(0~9)
# (3) validation_split=20: 20%的訓練數據將不用於訓練模型，而是評估模型在未見過的數據表現。

# 模型評估
model.evaluate(x_test, y_test)
# model.evaluate(): 測量模型的準確度或其他在模型編譯時指定的指標
# x_test: 數據在訓練過程中未被看見
# y_test: x_test的真實對應標籤(0~9)
# 輸出: 最終返回損失值和測試準確度(因model.compile(metrics=['accuracy']))
