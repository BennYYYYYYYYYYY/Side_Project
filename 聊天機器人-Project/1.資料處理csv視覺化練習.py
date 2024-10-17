import csv 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

df = pd.read_csv(r'C:\Users\user\Desktop\Python\Demo\marriage.csv')

data = df.loc[:, ['月份區域別', '一月']] # df.loc[row, ['columns_name']]
# print(data)
data = data.set_index('月份區域別') # 將指定的列設置為 DataFrame 的索引
# print(data)

# fig = data.plot(kind='line')
axes = data.plot(kind='bar')

plt.title('一月份各區域結婚數')
plt.xlabel('區域別')
plt.ylabel('結婚數')
plt.legend() # show 出圖例 
plt.show()

