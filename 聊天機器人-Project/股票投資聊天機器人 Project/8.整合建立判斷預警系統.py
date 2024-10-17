from apscheduler.schedulers.blocking import BlockingScheduler

# 設計一個定時執行程式在週間 9-14 整點，每小時執行一次
sched = BlockingScheduler()
@sched.scheduled_job('cron', day_of_week='mon-fri', hour='9-14')
def scheduled_job():
    print('每週週間 9am-14pm UTC+8. 執行此程式工作區塊!')


'''
1. 因為要確認目前股價的區間，所以設計一個函式進行條件判斷，傳入的參數有：
    stock_no、high_price、middle_price、low_price 和 latest_trade_price
        最後會回傳
            1. 【目前是屬於哪一個區間的字串】
            2. 【目前成交價格】
'''

def get_check_price_rule_message(stock_no, high_price, middle_price, low_price, latest_trade_price):
    """
    1. 目前股價太貴了：成交價 > 昂貴價
    2. 目前股價介於昂貴價和合理價之間：昂貴價 > 成交價 > 合理價
    3. 目前股價介於合理價和便宜價之間：合理價 > 成交價 > 便宜價
    4. 目前股價很便宜：便宜價 > 成交價
    """
    if latest_trade_price > high_price:
        message_str = f'{stock_no}:目前股價太貴了({latest_trade_price})'
    elif high_price > latest_trade_price and latest_trade_price > middle_price:
        message_str = f'{stock_no}:目前股價介於昂貴價和合理價之間({latest_trade_price})'
    elif middle_price > latest_trade_price and latest_trade_price > low_price:
        message_str = f'{stock_no}:目前股價介於合理價和便宜價之間({latest_trade_price})'
    elif low_price > latest_trade_price:
        message_str = f'{stock_no}:目前股價很便宜({latest_trade_price})'

    return message_str


'''
2. 設計一個定時執行程式在週間 9-14 整點，每小時執行一次 (測試時可以改為每 30 秒執行一次)
'''

from apscheduler.schedulers.blocking import BlockingScheduler
import twstock
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from linebot import LineBotApi
from linebot.models import TextSendMessage

# 設定 Google 試算表憑證
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('path/to/credentials.json', scope)
client = gspread.authorize(creds)
worksheet = client.open("你的試算表名稱").sheet1

# 設定 LINE Bot API
LINE_USER_ID = '你的LINE_USER_ID'
line_bot_api = LineBotApi('你的LINE_CHANNEL_ACCESS_TOKEN')

def get_check_price_rule_message(stock_no, high_price, middle_price, low_price, latest_trade_price):
    # 根據你的邏輯定義價格規則訊息
    pass

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

sched = BlockingScheduler()
sched.start()