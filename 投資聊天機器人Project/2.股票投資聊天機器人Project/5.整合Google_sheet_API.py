import os
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials 
import requests
import pandas as pd
from bs4 import BeautifulSoup 
from apscheduler.schedulers.blocking import BlockingScheduler


sched = BlockingScheduler() # 創建一個 Scheduler 物件實例

# 使用 Google API 的範圍為 spreadsheets
gsp_scopes = ['https://spreadsheets.google.com/feeds'] 
# ['https://spreadsheets.google.com/feeds']: 這是一個 URL，代表 Google Sheets API 的讀寫權限


# 使用 os 模組的 environ.get 方法從環境變數中獲取 Google Sheets 的鍵 (Spreadsheet Key)，用於識別要操作的 Google Sheets
SPREAD_SHEETS_KEY = os.environ.get('SPREAD_SHEETS_KEY')

# 金鑰檔案路徑
credential_file_path = r''

# auth_gsp_client 函數用來創建並返回一個經過認證的 gspread 客戶端對象
def auth_gsp_client(file_path, scopes):
    # 從檔案讀取金鑰資料
    
    credentials = ServiceAccountCredentials.from_json_keyfile_name(file_path, scopes) 
    # 使用 ServiceAccountCredentials 類的 from_json_keyfile_name 方法，從指定的 JSON 文件中讀取憑證，並使用這些憑證【創建一個憑證對象】
    
    return gspread.authorize(credentials) # 使用 gspread 庫的 authorize 方法，根據憑證對象進行授權
    # 並返回一個授權的 gspread 客戶端對象

# 使用 auth_gsp_client 函數並傳入金鑰文件路徑和權限範圍，獲得經授權的 gspread 客戶端對象
gsp_client = auth_gsp_client(credential_file_path, gsp_scopes)

# 使用 open_by_key 方法根據 SPREAD_SHEETS_KEY 打開 Google Sheets，並選擇第一個工作表 (sheet1)
worksheet = gsp_client.open_by_key(SPREAD_SHEETS_KEY).sheet1

def get_stock_price():
    url = 'https://goodinfo.tw/tw/StockDividendPolicy.asp?STOCK_ID=2330'
    headers = {
        'user-agent':'',
        'cookie':''
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

    high_price = cash*30
    middle_price = cash*20
    low_price = cash*15

    print('開始寫入資料...')
    # 使用 insert_row 方法將 [high_price, middle_price, low_price] 這三欄數據插入到第二列
    worksheet.insert_row([high_price, middle_price, low_price], 2) 
    print('成功寫入資料...')


#    print(f'便宜價:{cash*15}\n合理價:{cash*20}\n昂貴價:{cash*30}')

#    with open('stock_price.csv', 'w', encoding = 'utf-8-sig') as file:
#        file.write(f'便宜價,合理價,昂貴價\n')
#        file.write(f'{cash*15},{cash*20},{cash*30}\n')

# decorator 設定 Scheduler 的類型和參數，例如 interval 間隔多久執行
@sched.scheduled_job('interval', minutes=5)
def crawl_for_stock_price_job():

    # 要注意不要太頻繁抓取，但測試時可以調整時間少一點方便測試
    print('每 5 分鐘執行一次程式工作區塊')

    # 每次清除之前資料
    worksheet.clear()

    # 將標頭插入第 1 列
    print('開始寫入標頭...')

    worksheet.insert_row(['high_price', 'middle_price', 'low_price'], 1)
    print('成功寫入標頭...')

    # 再跑一次爬蟲，複寫資料
    get_stock_price()

# 開始執行
sched.start()


# 讀取 Google sheet 資料
'''
1. 取出第一列的值
    values_list = worksheet.row_values(1)

2. 取出第一欄的值
    values_list = worksheet.col_values(1)

3. 取出所有值
    list_of_lists = worksheet.get_all_values()
'''
