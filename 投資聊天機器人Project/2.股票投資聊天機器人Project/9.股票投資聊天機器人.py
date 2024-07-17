'''
最後就可以在本機電腦端進行測試，沒問題後透過 git 部屬到 heroku 上

    注意：
        定期執行的排程會依照設定的時間週期執行，並不會馬上執行，若要開發測試，可以改成比較短的時間來測試
'''

import os  # 多種操作系統相關功能，如文件和目錄操作
import json # 用於處理 JSON 資料的編碼和解碼

import pandas as pd # 強大的資料分析工具，用於處理數據框和其他數據結構

import gspread # 用於操作 Google 試算表
from oauth2client.service_account import ServiceAccountCredentials # 用於處理 Google API 認證

import requests # 用於進行 HTTP 請求
from bs4 import BeautifulSoup # 用於解析 HTML 和 XML 文件

from apscheduler.schedulers.blocking import BlockingScheduler #  引用 BlockingScheduler 類別，用於定時任務調度

import twstock # 用於取得台灣股票市場的即時數據

from linebot import (  # 用於與 LINE Bot API 進行交互
    LineBotApi
)
from linebot.models import (  # 用於發送文字訊息
    TextSendMessage,
)


# 創建一個 Scheduler 物件實例
sched = BlockingScheduler()

# 我們使用 Google API 的範圍為 spreadsheets
gsp_scopes = ['https://spreadsheets.google.com/feeds']
SPREAD_SHEETS_KEY = os.environ.get('SPREAD_SHEETS_KEY')

# LINE Chatbot token
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_USER_ID = os.environ.get('LINE_USER_ID')
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)

# 金鑰檔案路徑
credential_file_path = 'credentials.json'

# auth_gsp_client 為我們建立來產生金鑰認證物件回傳給操作 Google Sheet 的客戶端 Client
def auth_gsp_client(file_path, scopes):
    # 從檔案讀取金鑰資料
    credentials = ServiceAccountCredentials.from_json_keyfile_name(file_path, scopes)

    return gspread.authorize(credentials)


gsp_client = auth_gsp_client(credential_file_path, gsp_scopes)
# 我們透過 open_by_key 這個方法來開啟工作表一 worksheet
worksheet = gsp_client.open_by_key(SPREAD_SHEETS_KEY).sheet1

def get_stock_price(stock_no):
    url = f'https://goodinfo.tw/tw/StockDividendPolicy.asp?STOCK_ID={stock_no}'
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

    int_cash_list = []

    for i in cash_list:
        int_data = float(i)
        int_cash_list.append(int_data)

    cash = pd.Series(int_cash_list)
    cash = cash.mean() 

    high_price = cash*30
    middle_price = cash*20
    low_price = cash*15

    print('開始寫入資料...')
    worksheet.insert_row([stock_no, high_price, middle_price, low_price], 2) 
    print('成功寫入資料...')

# decorator 設定 Scheduler 的類型和參數，例如 interval 間隔多久執行
@sched.scheduled_job('interval', days=10)
def crawl_for_stock_price_job():
    # 要注意不要太頻繁抓取
    print('每 5 分鐘執行一次程式工作區塊')
    # 每次清除之前資料
    worksheet.clear()
    # 將標頭插入第 1 列
    print('開始寫入標頭...')
    worksheet.insert_row(['stock_no', 'high_price', 'middle_price', 'low_price'], 1)
    print('成功寫入標頭...')
    sotck_no_list = ['2330']
    # 第一筆資料股票代號
    get_stock_price(sotck_no_list[0])


def get_check_price_rule_message(stock_no, high_price, middle_price, low_price, latest_trade_price):
    if latest_trade_price > high_price:
        message_str = f'{stock_no}:目前股價太貴了({latest_trade_price})'
    elif high_price > latest_trade_price and latest_trade_price > middle_price:
        message_str = f'{stock_no}:目前股價介於昂貴價和合理價之間({latest_trade_price})'
    elif middle_price > latest_trade_price and latest_trade_price > low_price:
        message_str = f'{stock_no}:目前股價介於合理價和便宜價之間({latest_trade_price})'
    elif low_price > latest_trade_price:
        message_str = f'{stock_no}:目前股價很便宜({latest_trade_price})'

    return message_str


# 設計一個定時執行程式在週間 9-14 整點，每小時執行一次
@sched.scheduled_job('cron', day_of_week='mon-fri', hour='9-14')
def get_notify():
    print('開始讀取資料')
    stock_item_lists = worksheet.get_all_values()
    # 目前以一檔股票為範例
    sotck_no_list = ['2330']
    for stock_item in stock_item_lists:
        stock_no = stock_item[0]
        high_price = stock_item[1]
        middle_price = stock_item[2]
        low_price = stock_item[3]
        if str(stock_no) in sotck_no_list:
            # 擷取即時成交價格
            latest_trade_price = twstock.realtime.get(stock_no)['realtime']['latest_trade_price']
            price_rule_message = get_check_price_rule_message(stock_no, high_price, middle_price, low_price, latest_trade_price)
            line_bot_api.push_message(
                to=LINE_USER_ID,
                messages=[
                    TextSendMessage(text=price_rule_message)
                ]
            )
    print('通知結束')

# 開始執行
sched.start()
