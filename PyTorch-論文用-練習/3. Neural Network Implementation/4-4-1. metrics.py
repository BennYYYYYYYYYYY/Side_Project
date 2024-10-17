# 載入套件
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix

'''
(範例一)
混淆矩陣(Confusion Matrix)
'''
y_true = [0, 0, 0, 1, 1, 1, 1, 1] # 實際值
y_pred = [0, 1, 0, 1, 0, 1, 0, 1] # 預測值
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # ravel(): 攤平數據, 因為混淆矩陣是2x2矩陣, 所以需要攤平
print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')
# 真陽性（True Positive, TP）
# 假陽性（False Positive, FP）
# 真陰性（True Negative, TN）
# 假陰性（False Negative, FN）

# 繪圖
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 指定使用微軟正黑體
plt.rcParams['axes.unicode_minus']=False # 負號使用減號

# 顯示矩陣
fig, ax = plt.subplots(figsize=(2.5, 2.5)) # plt.subplots(): 創造一個圖表與子圖, 在這創造2.5*2.5的圖表
# fig: 整個圖的物件, ax: 圖表中子圖的物件 

ax.matshow([[1, 0], [0, 1]], cmap=plt.cm.Blues, alpha=0.3) # 1:藍色, 0:白色
# matshow(): 用於將矩陣或二維數組顯示為顏色編碼的圖像。
# [[1, 0], [0, 1]]: 要顯示的矩陣
# plt.cm.Blues: 提供顏色漸變的效果
# alpha: 透明度, 0(透明), 1(不透明)

# 指定位置添加文本
ax.text(x=0, y=0, s=tp, va='center', ha='center') # 指定座標(0,0)
# s=tp; 指定要顯示的文本字符串
# va垂直對齊(vertical alignment): 這裡設為center代表文本在垂直方向上將位於指定坐標的中心
# ha水平對其(horizontal alignment): 在水平方向上將位於指定坐標的中心
ax.text(x=1, y=0, s=fp, va='center', ha='center')
ax.text(x=0, y=1, s=tn, va='center', ha='center')
ax.text(x=1, y=1, s=fn, va='center', ha='center')

plt.xlabel('實際', fontsize=20) # x軸字
plt.ylabel('預測', fontsize=20) # y軸字

plt.xticks([0, 1], ['T', 'F']) # T對應位置0, F對應位置1, x軸有兩個刻度標籤，分別是'T'和'F'
plt.yticks([0, 1], ['P', 'N']) # P對應位置0, N對應位置1, y軸有兩個刻度標籤，分別是'P'和'N'
plt.show()



'''
(範例2)
準確率Accuracy
'''
print(f'準確率:{accuracy_score(y_true, y_pred)}')
print(f'驗算={(tp+tn) / (tp+tn+fp+fn)}')  


'''
(範例3)
精確率Precision
'''
print(f'精確率:{precision_score(y_true, y_pred)}')
print(f'驗算={(tp) / (tp+fp)}')  


'''
(範例4)
召回率recall
'''
print(f'召回率:{recall_score(y_true, y_pred)}')
print(f'驗算={(tp) / (tp+fn)}')  


'''
(範例5)
依資料檔data/auc_data.csv計算AUC

fpr作為x軸，tpr作為y軸。ROC曲線展示了在不同閾值下模型識別正類和負類的能力。
AUC（ROC曲線下的面積）則用於量化模型的整體性能，AUC值越高，表示模型的分類性能越好。
'''
import pandas as pd
df = pd.read_csv('C:\\Users\\user\\Desktop\\Python\\pytorch data\\data\\auc_data.csv')
print(df)

# 以sklearnd函數計算AUC
from sklearn.metrics import roc_curve, roc_auc_score, auc

# fpr:假陽率, tpr:真陽率, threshold: 數組，包含用於計算fpr和tpr的閾值
fpr, tpr, threshold = roc_curve(df['actual'], df['predict']) # 從DataFrame中取出真實標籤、預測結果
print(f'假陽率={fpr}\n\n真陽率={tpr}\n\n決策門檻={threshold}')

# 繪製AUC
auc1 = auc(fpr, tpr)
plt.title('ROC/AUC')
plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc1) # label: 添加說明文本圖例, %插入auc1的值, 並計算到小數後第2位
plt.legend(loc = 'lower right') # 顯示圖例(因為有label), 位置放右下
plt.plot([0, 1], [0, 1], 'r--') # x, y座標, 線: 紅色虛線
plt.xlim([0, 1]) # x軸顯示範圍
plt.ylim([0, 1]) # y軸顯示範圍
plt.ylabel('True Positive Rate') 
plt.xlabel('False Positive Rate')
plt.show()   
