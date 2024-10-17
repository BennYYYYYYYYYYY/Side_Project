'''
1. 整合 Google Sheets API :
    可以透過聊天機器人來紀錄外匯交易和損益試算查詢

2. 專案主要功能: 
    1. 讓使用者可以查詢匯率資訊
    2. 讓使用者可以紀錄外匯交易資料、查詢目前損益 

3. Google Sheets 試算表設定:
    1. 試算表加入標頭
        【交易日期, 交易幣別, 買進賣出, 交易單位, 成交金額】

4. 整合聊天機器人輸入:
    希望讓使用者可以輸入格式
        例如：買/USD/2000 就可以紀錄該筆外幣買賣紀錄

5. 建立計算函式:
    透過 twder 取得該幣別匯率資訊並根據買或賣來決定單位價格 

6. 查詢計算損益:
    由於 reply_message API 回傳有時間限制，若計算太久會來不及回傳
    所以需要另外使用 push_message API，Basic Settings中有有 User ID
        同樣把 User ID 存到電腦環境變數

7. 整合聊天機器人輸入:
    讓使用者可以查詢損益
        當使用者輸入【@查詢損益】時(也可以使用圖文選單)
        會計算目前試算表中的交易總成本和目前查詢的匯率價格所計算的獲利的損益。
            由於計算時間有可能超過 reply_message 回覆期間 reply_token 有可能會過期，所以使用 push_message API

8. 建立計算損益函式:
    建立一個計算損益函式 get_currency_net_profit 來計算試算表中交易資料的損益
        【計算淨利：現在賣出的價格(即期買入) * 剩餘貨幣數量 - 交易總成本】

9. 整合外幣管理聊天機器人專案:
    完成了專案程式後可以利用 Flask 本地端 Server 搭配 ngrok 進行本地端測試。
    同樣測試沒問題後可以推送到雲端上，用Line測試
'''
import os

# 引入套件 flask
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
# 引入 linebot 異常處理
from linebot.exceptions import (
    InvalidSignatureError
)
# 引入 linebot 訊息元件
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
# 引用查詢匯率套件
import twder

# Spread Sheets API 套件
import gspread
from oauth2client.service_account import ServiceAccountCredentials 

app = Flask(__name__)

# 我們使用 Google API 的範圍為 spreadsheets
gsp_scopes = ['https://spreadsheets.google.com/feeds']
GOOGLE_SHEETS_CREDS_JSON = os.environ.get('GOOGLE_SHEETS_CREDS_JSON')
SPREAD_SHEETS_KEY = os.environ.get('SPREAD_SHEETS_KEY')

# LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 類似聊天機器人的密碼，記得不要放到 repl.it 或是和他人分享
# 從環境變數取出設定參數
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
LINE_USER_ID = os.environ.get('LINE_USER_ID')
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def get_all_currencies_rates_str():
    """取得所有幣別目前匯率字串
    """
    all_currencies_rates_str = '' # 初始化一個空字串，用來存放最終的所有幣別匯率字串
    # all_currencies_rates 是一個 dict
    all_currencies_rates = twder.now_all() 
    '''
    twder.now_all() 函數會回傳一個包含所有幣別目前匯率資訊的字典
    
    {
        '幣別': (時間戳記, 現金買入價, 現金賣出價, 即期買入價, 即期賣出價)
        'USD': ('2024-07-14 12:34:56', '30.000', '30.500', '30.200', '30.400'),
        'EUR': ('2024-07-14 12:34:56', '33.500', '34.000', '33.700', '33.900'),
        'JPY': ('2024-07-14 12:34:56', '0.270', '0.275', '0.272', '0.274'),
        ...
    }

    '''
    # 遍歷字典中的每一對 key-value。key 是貨幣代碼，value 是一個包含匯率資訊的 tuple。
    for currency_code, currency_rates in all_currencies_rates.items():
        # \ 為多行斷行符號，避免組成字串過長
        all_currencies_rates_str += f'[{currency_code}] 現金買入:{currency_rates[1]} \
            現金賣出:{currency_rates[2]} 即期買入:{currency_rates[3]} 即期賣出:{currency_rates[4]} ({currency_rates[0]})\n'
    return all_currencies_rates_str


# 金鑰檔案路徑
credential_file_path = 'credentials.json'

# auth_gsp_client 為我們建立來產生金鑰認證物件回傳給操作 Google Sheet 的客戶端 Client
def auth_gsp_client(file_path, scopes):
    # 從檔案讀取金鑰資料
    credentials = ServiceAccountCredentials.from_json_keyfile_name(file_path, scopes)

    return gspread.authorize(credentials)


gsp_client = auth_gsp_client(credential_file_path, gsp_scopes)
# 我們透過 open_by_key 這個方法來開啟工作表一 worksheet
worksheet = gsp_client.open_by_key(SPREAD_SHEETS_KEY).worksheet('transaction')


def record_currency_transaction(action, currency, unit):
    """紀錄交易
    action: 買/賣
    currency: ['CNY', 'THB', 'SEK', 'USD', 'IDR', 'AUD', 'NZD', 'PHP', 'MYR', 'GBP', 'ZAR', 'CHF', 'VND', 'EUR', 'KRW', 'SGD', 'JPY', 'CAD', 'HKD']
    unit: 數量
    """
    current_row_length = len(worksheet.get_all_values()) # worksheet.get_all_values() 會回傳試算表中所有的數據
    currency_data = twder.now(currency) # 取得指定貨幣(key)的當前匯率資訊

    # currency_data[0] 包含日期和時間，使用 split(' ') 方法將其分割，並取出第一部分(日期)
    # 例如 '2024-07-14 12:00:00'。split(' ') 方法會將這個字串分割成 ['2024-07-14', '12:00:00']
    transaction_date = currency_data[0].split(' ')[0]

    # 如果 action 是 '買'，則使用即期賣出的匯率 (currency_data[4])
    if action == '買':
        currency_price = currency_data[4]

    # 如果 action 是 '賣'，則使用即期買入的匯率 (currency_data[3])
    elif action == '賣':
        currency_price = currency_data[3]

    # 寫入試算表欄位：交易日期, 交易幣別, 買進賣出, 交易單位, 成交金額
    worksheet.insert_row([transaction_date, currency, action, unit, currency_price], current_row_length + 1)
    # insert_row(想要插入的每個單元格的值, 指定要插入的位置(從1開始)) 

    return True # 告知函數的調用者這次操作是否成功完成


def get_currency_net_profit():
    records = worksheet.get_all_values() # 回傳試算表中的所有數據
    currency_statistics_data = {}
    print('計算中...')
    for index, record in enumerate(records):
        # 如果索引值為 0，表示這是標頭行，跳過這行
        if index == 0:
            continue
        currency = record[1] # 記錄中提取貨幣代碼
        action = record[2] # 交易動作
        unit = float(record[3]) # 交易單位
        price = float(record[4]) # 成交金額
        
        # 計算交易成本
        cost = unit * price

        # 如果 currency_statistics_data 中沒有這種貨幣的資料，則為它初始化一個新的字典，並設置 total_cost 和 total_unit 為 0
        if currency not in currency_statistics_data:
            currency_statistics_data[currency] = {} # currency_statistics_data 中沒有該貨幣的資料，則新增一個空字典作為該貨幣的統計資料
            currency_statistics_data[currency]['total_cost'] = 0 # 第一層key:[currency]，第二層key['total_cost']
            currency_statistics_data[currency]['total_unit'] = 0
        
        # 如果是買，增加總成本和總單位數量
        if action == '買':
            currency_statistics_data[currency]['total_cost'] += cost
            currency_statistics_data[currency]['total_unit'] += unit
  
        # 如果是賣，減少總成本和總單位數量
        elif action == '賣':
            currency_statistics_data[currency]['total_cost'] -= cost
            currency_statistics_data[currency]['total_unit'] -= unit

        '''
            currency_statistics_data = { 
                'USD': {
                    'total_cost': 15000.0,
                    'total_unit': 500.0
                },
                'JPY': {
                    'total_cost': 2600.0,
                    'total_unit': 10000.0
                },
                'EUR': {
                    'total_cost': 35250.0,
                    'total_unit': 1000.0
                }
            }

        '''


    currency_net_profit_str = '' # 初始化空字串，用於存放最終的淨利結果


    for currency, currency_data in currency_statistics_data.items(): # items_view 是一個包含字典中所有鍵值對的視圖對象
        # currency 是貨幣代碼(key)，例如'USD'、'JPY' 
        # currency_data 是key的所有統計資料(value)，包含 total_cost 和 total_unit

        # 使用 twder.now(currency) 函數獲取指定貨幣的當前匯率資料
        now_currency_data = twder.now(currency)
        # 例如：('2024-07-14 12:34:56', '30.000', '30.500', '30.200', '30.400')


        # now_currency_data[3] 是即期買入價。如果這個值不是 '-'，則表示這個價格是有效的，可以進行後續計算
        if now_currency_data[3] != '-':
            current_price = float(now_currency_data[3])
            net_profit = current_price * currency_data['total_unit'] - currency_data['total_cost']
            currency_net_profit_str += f'[{currency}]損益:{net_profit:.2f}\n' # 將 net_profit 格式化為兩位小數

    return currency_net_profit_str # 回傳包含所有貨幣淨利潤結果的字串 currency_net_profit_str


# 此為歡迎畫面處理函式，當網址後面是 / 時由它處理
@app.route("/", methods=['GET'])
def hello():
    return 'hello heroku'


# 此為 Webhook callback endpoint 處理函式，當網址後面是 /callback 時由它處理
@app.route("/callback", methods=['POST'])
def callback():
    # 取得網路請求的標頭 X-Line-Signature 內容，確認請求是從 LINE Server 送來的
    signature = request.headers['X-Line-Signature']

    # 將請求內容取出
    body = request.get_data(as_text=True)

    # handle webhook body（轉送給負責處理的 handler，ex. handle_message）
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


# 使用裝飾器 @handler.add 來註冊一個訊息處理函數 handle_message
@handler.add(MessageEvent, message=TextMessage) # MessageEvent 表示這個處理函數會處理來自 LINE 的訊息事件
# TextMessage 表示這個處理函數專門處理文字訊息

def handle_message(event): # 處理來自 LINE 的文字訊息的函數
    user_input = event.message.text # event 是一個 MessageEvent 對象，包含了關於這個事件的所有信息
   
    # 如果用戶輸入的訊息是 @查詢所有匯率，則調用 get_all_currencies_rates_str() 函數獲取所有幣別的匯率資訊
    if user_input == '@查詢所有匯率':
        all_currencies_rates_str = get_all_currencies_rates_str()
        # # 使用 line_bot_api.reply_message 方法回覆用戶，將 all_currencies_rates_str 作為回覆訊息發送
        line_bot_api.reply_message( 
            event.reply_token,
            TextSendMessage(text=all_currencies_rates_str))
        

    # 如果用戶輸入的訊息包含 '買/' 或 '賣/'，則表示用戶要紀錄一筆交易
    elif '買/' in user_input or '賣/' in user_input:
        # 使用 split('/') 方法將用戶輸入的訊息分割成三部分：交易動作（買或賣）、貨幣代碼和數量
        split_user_input = user_input.split('/')
        action = split_user_input[0]
        currency = split_user_input[1]
        unit = split_user_input[2]

        # 調用 record_currency_transaction(action, currency, unit) 函數來紀錄這筆交易
        record_currency_transaction(action, currency, unit)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='紀錄完成')) # 通知用戶交易已被紀錄
        

    # 果用戶輸入的訊息是 @查詢損益，則首先回覆用戶 "計算中"
    elif user_input == '@查詢損益':
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='計算中'))
        # 調用 get_currency_net_profit() 函數來計算所有交易的損益
        currency_net_profit = get_currency_net_profit()

        # 由於計算可能耗時較長，所以使用 push_message 而不是 reply_message
        line_bot_api.push_message(
            LINE_USER_ID,
            TextSendMessage(text=currency_net_profit))

# __name__ 為內建變數，若程式不是被當作模組引入則為 __main__
if __name__ == '__main__':
    app.run()