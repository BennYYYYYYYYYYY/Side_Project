'''
(問題4)
優化器、損失函數、效能指標函數: 有哪些選擇? 設為其他值有什麼影響?

優化器optimizer: 固定值Lr、動態Lr、自訂Lr
損失函數loss: 常見(MSE、CrossEntropy)、特殊功能(風格轉換StyleTransfer、生成對抗網路GAN)
效能指標函數metrics: 準確率Accuracy、精確率Precision、召回率Recall、F1
'''

'''
準確率Accuracy: 模型正確預測的樣本數占總樣本數的比例
精確率Precision: 所有被模型預測為正例的樣本中，實際為正例的比例
召回率Recall: 在所有實際為正例的樣本中，被模型正確預測為正例的比例
F1 Score: 精確率和召回率的調和平均值, 當精確率和召回率都高時，F1分數也會高
'''