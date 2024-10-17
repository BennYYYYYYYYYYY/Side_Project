'''
(一) 歷年股價估價法：
歷年股價估價法為參考該公司過去 5 年的歷年股價區間，找出最高價（昂貴價）和最低價（便宜價），中間值（最高和最低平均）

(二) 本益比估價法：
本益比(PE Ratio)：目前股票市場價格 / 每股盈餘 (EPS = (稅後淨利 - 當年特別股股利)／加權平均流通在外的普通股股數)
透過參考公司過去 5 年的最高本益比和最低本益比搭配目前今年預估的 EPS 每股盈餘，
例如：(最近四季累積 EPS + 最近 5 年平均 EPS) / 2 或根據公開新聞推估去年累計 EPS 就可以使用本益比估價法推估股價區間

(三) 平均現金殖利率估價法 (平均股利估價法)：
殖利率 是一般評估公司股利發放程度的重要指標，其參考公式為：現金殖利率 = (現金股息 ÷ 股價) × 100%。
所以我們可以透過過去 5 年最高殖利率和最低殖利率搭配來估算股價區間。
另外一種簡單方式為：
過去 5 或 10 年平均股利 * 15 為便宜價，
過去 5 或 10 年平均股利 * 20 為合理價，
當期 過去 5 或 10 年平均股利 * 30 為昂貴價。

(四) 混合平均法：
混合平均法是將多種估價法結合起來，以獲得更準確的股價區間。
常見的混合方法包括歷年股價估價法、本益比估價法、平均現金殖利率估價法等的加權平均。
1. 使用歷年股價估價法計算股價區間。
2. 使用本益比估價法計算股價區間。
3. 使用平均現金殖利率估價法計算股價區間。
4. 將三者的估計結果加權平均（根據你對各方法可靠性的評估給予不同的權重）。

(五) 股價淨值比估價法 (Price-to-Book Ratio, P/B Ratio)：
股價淨值比是一種用來衡量公司股價與其每股淨資產(Book Value)的比率，公式如下：
股票市場價格 / 每股淨資產，根據過去數年的股價淨值比範圍（最高與最低），搭配目前的每股淨資產，可以估算出股價範圍。例如：
便宜價 = 最低股價淨值比 \ 每股淨資產 (每股淨值：總資產減去總負債，再除以流通在外的普通股數)
一般來說，P/B 比率低於1被認為是低估，高於1.5可能被視為高估

'''

# Goodinfo 2330 個股K線圖
#row0 > td:nth-child(5) > nobr

import requests
from bs4 import BeautifulSoup

url = 'https://goodinfo.tw/tw/ShowK_Chart.asp?STOCK_ID=2330&CHT_CAT=YEAR&PRICE_ADJ=F&SCROLL2Y=600'

headers = {
    
}

res = requests.get(url, headers=headers)
res.encoding = 'utf-8'
# 根據 HTTP header 的編碼解碼後的內容資料（ex. UTF-8）

soup = BeautifulSoup(res.text, 'lxml')
price_rows = []
'''
# 使用選擇器選取最近五日，由於年份是由列決定，所以我們取1-5列
for row_line in range(3, 8):
    # select 會取出符合條件的元素，但為 list 故要取 0 index 取出元素
    
    # select_one(selector)：使用 CSS 選擇器查找並返回第一個匹配的元素，如果沒有匹配的元素，則返回 None
    # select(selector)：使用 CSS 選擇器查找並返回所有匹配的元素，以列表形式返回，如果沒有匹配的元素，則返回空列表 []
    price_rows.append(soup.select_one(f'#divPriceDetail > tr:nth-child({row_line}) > td:nth-child(6)'))

print('price_rows', price_rows)
'''
data = soup.select_one('#divPriceDetail')
for row_line in range(0, 5):
    # select 會取出符合條件的元素，但為 list 故要取 0 index 取出元素
    
    # select_one(selector)：使用 CSS 選擇器查找並返回第一個匹配的元素，如果沒有匹配的元素，則返回 None
    # select(selector)：使用 CSS 選擇器查找並返回所有匹配的元素，以列表形式返回，如果沒有匹配的元素，則返回空列表 []
   # price_rows.append(detail.select_one(f'tr#row{row_line}').text)
    b = data.select_one(f'tr#row{row_line}').text
    # print(b)
    c = b.split(' ')
    # print(c)
    data_year = str(c[1])
    closing_price = str(c[5])
    price_rows.append('年份: '+ data_year[:4]+' 收盤價: ' + closing_price)
for row in price_rows:
    print(row)

# max/min 內建 Python 函式取出最高價和最低價格
max_price = max(price_rows)
min_price = min(price_rows)

"""
price_rows ['316', '331', '225.5', '229.5', '181.5', '143']
max_price 331
min_price 143
"""
print('max_price', max_price)
print('min_price', min_price)

# =================================================================================================

'''
目標：使用 Goodinfo 股票資訊網，點選【本益比河流圖】選擇年線，擷取以年為單位的價格資料。
'''

# #row0 > td:nth-child(5)

import requests
from bs4 import BeautifulSoup

url = 'https://goodinfo.tw/tw/ShowK_ChartFlow.asp?RPT_CAT=PER&STOCK_ID=2330&CHT_CAT=YEAR&SCROLL2Y=525'

headers = {
  
}

res = requests.get(url, headers=headers)
res.encoding = 'utf-8'

# PE Ratio 簡寫 per
soup = BeautifulSoup(res.text, 'lxml')
# print(res.text)

per_rows = [] # 本益比 Price to Earing Ratio(倍)
eps_rows = [] # EPS(元)

data = soup.select_one('#divDetail')
# print(data)

for row_number in range(0, 10):
    eps_data = data.select_one(f'tr#row{row_number} > td:nth-child(5)').text # .text為存取文本 
    # print(eps_data) # 成功抓取【河流圖EPS(元)】資料
    eps_rows.append('EPS:' + str(eps_data))  
# print(eps_rows)

    per_data = data.select_one(f'tr#row{row_number} > td:nth-child(6)').text # .text為存取文本 
    per_rows.append('PER:' + str(per_data))

# print(eps_rows)
# print(per_rows)

# 取出最高本益比和最低本益比
max_eps = max(eps_rows)
min_eps = min(eps_rows)

# 取出最高本益比和最低本益比
max_per = max(per_rows)
min_per = min(per_rows)

print('max_eps', max_eps)
print('min_eps', min_eps)

print('max_per', max_per)
print('min_per', min_per)

# =================================================================================================

'''
平均現金殖利率估價法 (平均股利估價法)

殖利率 是一般評估公司股利發放程度的重要指標，其參考公式為：現金殖利率 = (現金股息 ÷ 股價) × 100%。
所以我們可以透過過去 5 年最高殖利率和最低殖利率搭配來估算股價區間。
另外一種簡單方式為：
過去 5 或 10 年平均股利 * 15 為便宜價，
過去 5 或 10 年平均股利 * 20 為合理價，
當期 過去 5 或 10 年平均股利 * 30 為昂貴價。
'''
import requests
import pandas as pd
from bs4 import BeautifulSoup 

url = 'https://goodinfo.tw/tw/StockDividendPolicy.asp?STOCK_ID=2330'
headers = {
    
}

res = requests.get(url, headers=headers)
res.encoding = 'utf-8'
soup = BeautifulSoup(res.text, 'lxml')

# print(res.text)
cash_list = []

row_data = soup.select_one('#divDetail')
# print(cash_data)
cash_data = row_data.select_one(f'tr#row1 > td:nth-child(4)').text
cash_list.append(cash_data)

for row_line in range(5, 26, 5):
    cash_data = row_data.select_one(f'tr#row{row_line} > td:nth-child(4)').text
    cash_list.append(cash_data)

for row_line in range(29, 33):
    cash_data = row_data.select_one(f'tr#row{row_line} > td:nth-child(4)').text
    cash_list.append(cash_data)

# print(cash_list) # 抓出 2024~2015 現金股利

# print(type(cash_list[0]))  資料型態'str'

int_cash_list = []

for i in cash_list:
    int_data = float(i)
    int_cash_list.append(int_data)

# print(int_cash_list) 改成 num 的 list

cash = pd.Series(int_cash_list)
cash = cash.mean() # 把十年的現金股利做平均

print(f'便宜價:{cash*15}\n合理價:{cash*20}\n昂貴價:{cash*30}')

with open('stock_price.csv', 'w', encoding = 'utf-8-sig') as file:
    file.write(f'便宜價,合理價,昂貴價\n')
    file.write(f'{cash*15},{cash*20},{cash*30}\n')


