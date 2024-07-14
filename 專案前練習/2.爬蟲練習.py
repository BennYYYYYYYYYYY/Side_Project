import requests
import bs4
import csv

url = 'https://www.ptt.cc/bbs/Stock/index7321.html'

headers = {
    'user_agent': ''
}

res = requests.get(url, headers=headers)
res.encoding = 'utf-8'
raw_html = res.text

soup = bs4.BeautifulSoup(raw_html, 'lxml')

title_list = []

for index in range(2, 18):
    print('index', index)
    title_dict = {}
    title_dict['title'] = soup.select(f'#main-container > div.r-list-container.action-bar-margin.bbs-screen > div:nth-child({index}) > div.title > a')[0].text
    title_list.append(title_dict)

headers = ['title']

with open('stock.csv', 'w', newline='', encoding='utf-8-sig') as output_file: # encoding 避免亂碼(vscode, csv), newline 避免空行
    dict_writer = csv.DictWriter(output_file, headers)
    dict_writer.writeheader()
    dict_writer.writerows(title_list)

'''
element 元素標籤選擇器：透過 HTML 元素標籤來選取
id 選擇器：透過 HTML 標籤的 id 屬性來選取，使用 #名稱 進行選取。同一頁面 id 為唯一
class 選擇器：透過 HTML 標籤的 class 屬性來選取，使用 .名稱 進行選取
* 通用選擇器和：選取所有元素
[] 屬性選擇器：使用 [attribute]，代表選取含有 attribute 屬性的元素。
> 直屬選擇器：只會選到元素內直屬第一代元素，下一代不會受影響
element:nth-child(N)：代表在同一層元素中的第 N 個，從 1 開始起算
'''

# =================================================================================================

import requests
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Cookie':'IS_TOUCH_DEVICE=F; SCREEN_SIZE=WIDTH=1536&HEIGHT=864; _ga=GA1.1.857801794.1719046812; CLIENT%5FID=20240622170013128%5F116%2E241%2E197%2E191; TW_STOCK_BROWSE_LIST=2330%7C2412; __gads=ID=2adea7b463a82bb7:T=1719046813:RT=1719078681:S=ALNI_MZxvO1AoJGo0GDnj2q3ceUTKkLb0g; __gpi=UID=00000e5b66e0a172:T=1719046813:RT=1719078681:S=ALNI_MYf4kBtrvDZ-vnFZp-RdINwa8zVUQ; __eoi=ID=da9f277121250aa4:T=1719046813:RT=1719078681:S=AA-AfjY-UPruuxD-fOlGNkeS3Hp_; _ga_0LP5MLQS7E=GS1.1.1719078681.2.1.1719078713.28.0.0; FCNEC=%5B%5B%22AKsRol-IgNJw43dsXFUf1vxZrCCA0YsRbsV5gv9mh4KpEh4WCgKrZAqtH81scblfO79294zQTBtgiu8mcKHOKV8RaB2k-c1GYQZWZIvoGWLhV-0SII72i7EhhSblza1FRRe-mb-sYyRadWiRJXq87G38jMSkRWAwsQ%3D%3D%22%5D%5D'
}

# 使用 network 中的 request URL
res = requests.get('https://goodinfo.tw/tw/StockBzPerformance.asp?STOCK_ID=2330', headers=headers)
res.encoding = 'utf-8'
# print(res.text)  # 未能自動轉址，確認瀏覽器的javascript是否正常運作 (需要+cookie)

from bs4 import BeautifulSoup
soup = BeautifulSoup(res.text, 'lxml')

# id = txtFinDetailData
data = soup.select_one('#txtFinDetailData') 
# print(data)

import pandas as pd
# .prettify() 是 BeautifulSoup 模組中的一個方法，用來將HTML或XML文件轉換為漂亮的格式
df = pd.read_html(data.prettify())
# print(len(df)) # 此時 df 為 DataFrame 列表

# print(df[0]) 整個DataFrame
# df = df.drop(0)  表示我們希望刪除的行標籤 (索引)

df = pd.DataFrame(df[0]) # 讓list的第一項, 變成dataframe

# print(df.head(1)) 第一列

with open('stock_data.csv', 'w', encoding='utf-8-sig', newline='') as file:
    df.to_csv(file, index=False)

df = pd.read_csv(r'C:\Users\user\Desktop\Python\證券投資分析&股票聊天機器人\上市櫃公司財報資料分析專案\stock_data.csv')

column_name = df.columns.tolist() # 利用tolist(), 讓欄位名的 pandas.index (.columns) 變成 list

data = df.loc[:, column_name[:3]] # 選取所有row, 前3 columns

print(data)

with open('stock2.csv', 'w', encoding='utf-8-sig', newline='') as files:
    data.to_csv(files, index=False)

                  
# =================================================================================

# pip install selenium==3.141.0

# 安裝瀏覽器驅動引擎 driver

# 當爬取的網頁為 JavaScript 網頁前端（非伺服器端）生成，需引入 selenium 套件來模擬瀏覽器載入網頁並跑完 JavaScript 程式才能取得資料
# 引入套件
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import requests
from bs4 import BeautifulSoup

# ./chromedriver.exe 為 chrome 瀏覽器驅動引擎檔案位置（注意 MacOS/Linux 沒有 .exe 副檔名），也可以使用絕對路徑，例如： C:\downloads\chromedriver.exe
driver = webdriver.Chrome('./chromedriver.exe')

url = 'https://goodinfo.tw/StockInfo/ShowSaleMonChart.asp?STOCK_ID=2330'

# 透過 selenium 發出網路請求
driver.get(url)
# 取出網頁整頁內容
page_content = driver.page_source
# 印出網頁標題
print(driver.title)
# 印內容出來看看
print(page_content)

# 將 selenium 取出整頁的 HTML 轉成 BeautifulSoup 物件
soup = BeautifulSoup(page_content, "html.parser")

# 使用 CSS Selector 選到對應的元素位置，取出裡面的值，nobr 非標準的 HTML 元素其實可以忽略只寫 #divDetail > table > tr:nth-child(5) > td:nth-child(1)
revenue_date = soup.select('#divDetail > table > tr:nth-child(5) > td:nth-child(1) > nobr')[0].text
final_price = soup.select('#divDetail > table > tr:nth-child(5) > td:nth-child(3) > nobr')[0].text
year_revenue = soup.select('#divDetail > table > tr:nth-child(5) > td:nth-child(11) > nobr')[0].text

print('revenue_date:', revenue_date, 'final_price:', final_price, 'year_revenue', year_revenue)

# 關閉瀏覽器
driver.quit()

# =================================================================================================

import csv

import requests
from bs4 import BeautifulSoup


url = 'https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID=2330'

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
resp = requests.get(url, headers=headers)

# 設定編碼為 utf-8 避免中文亂碼問題
resp.encoding = 'utf-8'

# 根據 HTTP header 的編碼解碼後的內容資料（ex. UTF-8），若該網站沒設定可能會有中文亂碼問題。所以通常會使用 resp.encoding 設定
raw_html = resp.text


# 將 HTML 轉成 BeautifulSoup 物件
soup = BeautifulSoup(raw_html, 'html.parser')

def parse_str_to_float(raw_value):
    return float(raw_value.replace(',', ''))

# 開始寫入檔案，把資料存放到 list 裡面
# 若是忘記 list/dict 用法可以回去複習一下
performance_list = []

# 使用 CSS Selector 選到對應的元素位置，取出裡面的值。根據表格結構內容從 tr 第五行開始，所以 index 從 5 開始，到 9 結束（10 - 1，range 結束不含最後值）
for index in range(5, 10):
    print('index', index)
    performance_dict = {}
    performance_dict['date'] = soup.select(f'#divDetail > table > tr:nth-child({index}) > td:nth-child(1) > nobr')[0].text
    performance_dict['final_price'] = soup.select(f'#divDetail > table > tr:nth-child({index}) > td:nth-child(3) > nobr')[0].text
    performance_dict['year_revenue'] = soup.select(f'#divDetail > table > tr:nth-child({index}) > td:nth-child(11) > nobr')[0].text
    # 每月資料寫入 list
    performance_list.append(performance_dict)

# CSV 檔案第一列標題會是 date, final_price, year_revenue，記得要和 dict 的 key 相同，不然會出現錯誤
headers = ['date', 'final_price', 'year_revenue']

# 使用檔案 with ... open 開啟寫入檔案模式，透過 csv 模組將資料寫入
with open('performance.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, headers)
    # 寫入標題
    dict_writer.writeheader()
    # 寫入值
    dict_writer.writerows(performance_list)


# =================================================================================================
