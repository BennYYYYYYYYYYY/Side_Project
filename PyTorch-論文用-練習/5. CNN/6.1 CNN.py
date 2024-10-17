'''
卷積神經網路簡介:

CNN引進卷積層(Convolution layer)先進行特徵萃取(Feature Extraction), 將像素轉為各種線條特徵, 再交給Linear辨識

積卷(Convolution)簡單說就是圖形抽樣化(Abstraction), 把不必要的資訊刪除, 例如色彩、背景等等, 因此model即可依據這些線條辨別出是人、是車or是其他東西

1. 輸入圖片, 若為彩色的, 每個彩色通道分別卷積再合併
2. 圖像經過Convolution Layer運算, 變成Feature Map, 另外通常會附加ReLU Activation Function
3. 卷積層後面會接一個Pooling, 做Data Sampling, 以降低參數個數, 避免model過大
4. 最後把Feature Map flatten, 交給完全連結層辨識

'''

