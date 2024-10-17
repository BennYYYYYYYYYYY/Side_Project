import os # 與操作系統交互的功能，如文件和目錄操作、環境變量訪問
import requests 
import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask, request, abort # Flask是一個輕量級的Web框架，用於構建Web應用程序
# request對象用於訪問HTTP請求數據
# abort函數用於中止請求並返回特定的HTTP狀態碼

from linebot import (
    LineBotApi, WebhookHandler # LineBotApi用於與LINE Messaging API進行交互
) # WebhookHandler用於處理從LINE伺服器接收的Webhook事件

from linebot.exceptions import (
    InvalidSignatureError # 導入InvalidSignatureError異常類，用於處理簽名驗證失敗的情況
)

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, # MessageEvent表示接收到的消息事件
) # TextMessage表示文本消息
# TextSendMessage表示要發送的文本消息
'''
MessageEvent(消息事件)
    在LINE Messaging API中，當用戶向LINE官方帳號發送消息時，會產生一個事件。
    這個事件被LINE伺服器捕捉並發送到我們的Webhook URL。
    在我們的應用程序中，需要處理這些事件，而MessageEvent就是用來表示這些事件的。
'''



# 創建了一個Flask應用程序實例。__name__參數將當前模塊的名稱傳遞給Flask應用，用於決定應用根目錄
app = Flask(__name__)


headers = {
    'user-agent': ''
}


LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN') # 從環境變量中獲取LINE_CHANNEL_ACCESS_TOKEN
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET') # 從環境變量中獲取LINE_CHANNEL_SECRET
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

fund_map_dict = {} # 將用來存儲基金名稱和對應的基金組ID


def init_fund_list():
    resp = requests.get('https://www.sitca.org.tw/ROC/Industry/IN2421.aspx?txtMonth=02&txtYear=2020', headers=headers)
    soup = BeautifulSoup(resp.text, 'lxml')

    table_content = soup.select('#ctl00_ContentPlaceHolder1_TableClassList')[0] # id = 'ctl00_ContentPlaceHolder1_TableClassList'

    # <a>標籤通常用來表示超連結
    fund_links = table_content.select('a') # 這行代碼從選定的表格內容中選擇所有<a>標籤。

    for fund_link in fund_links: # 對每個超連結進行處理
        if fund_link.text: # 檢查fund_link.text是否存在 (鏈接是否有文本)
            fund_name = fund_link.text # 如果存在，則提取文本作為 fund_name
            fund_group_id = fund_link['href'].split('txtGROUPID=')[1] # 從鏈接的 href 屬性中提取基金組ID
            # 通過split方法分割href屬性字符串並取第二部分([1])

            fund_map_dict[fund_name] = fund_group_id # fund_name作為key，fund_group_id作為value
'''
<a> 範例:
    <a href="some_url?txtGROUPID=12345">基金A</a>
    <a href="some_url?txtGROUPID=67890">基金B</a>

對於第一個<a>標籤，fund_link.text會是"基金A"，fund_link['href']會是"some_url?txtGROUPID=12345"。
對於第二個<a>標籤，fund_link.text會是"基金B"，fund_link['href']會是"some_url?txtGROUPID=67890"。

    <html>
    <head><title>Example Page</title></head>
    <body>
    <p class="title"><b>Example Page</b></p>
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
    <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
    <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    <p class="story">...</p>
    </body>
    </html>

soup = BeautifulSoup(html_doc, 'lxml')

1. 選擇具有 href 屬性的 <a> 標籤
    links_with_href = soup.select('a[href]')

2. 選擇 href 屬性包含 "example" 的 <a> 標籤
    example_links = soup.select('a[href*="example"]')

3. 選擇 class 屬性等於 "sister" 的 <a> 標籤
    sister_links = soup.select('a[class="sister"]')

'''

def fetch_fund_rule_items(year, month, group_id):
    fetch_url = f'https://www.sitca.org.tw/ROC/Industry/IN2422.aspx?txtYEAR={year}&txtMONTH={month}&txtGROUPID={group_id}'
#   print(year, month, group_id, fetch_url)
    resp = requests.get(fetch_url, headers=headers)

    soup = BeautifulSoup(resp.text, 'lxml')

    # 選擇ID為ctl00_ContentPlaceHolder1_TableClassList的HTML元素，並取其第一個匹配項。
    table_content = soup.select('#ctl00_ContentPlaceHolder1_TableClassList')[0]


    fund_df = pd.read_html(table_content.prettify(), encoding='utf-8')[1] # 選擇第二個表格
    # 使用pandas的read_html方法從表格HTML中提取數據，並將其轉換為DataFrame對象
    # prettify方法將HTML格式化為易讀的形式

    fund_df = fund_df.drop(index=[0]) # index=[0]: 刪除DataFrame中的第一行(標題)
    fund_df.columns = fund_df.iloc[0] # 將DataFrame的列名設置為原來的第二行, iloc[行,列]
    fund_df = fund_df.drop(index=[1]) # 這行代碼刪除新的標題行 (原來的第二行，現在的索引為1)
    fund_df.reset_index(drop=True, inplace=True)  # 重置DataFrame的索引，並將原來的索引丟棄 (drop=True)
    # inplace=True: 表示直接在原DataFrame上進行操作

    fund_df = fund_df.fillna(value=0) # fillna: 用於填充缺失值。value=0: 所有的缺失值都被替換為0

    # 將DataFrame中指定列的數據類型轉換為浮點數
    fund_df['一個月'] = fund_df['一個月'].astype(float)
    fund_df['三個月'] = fund_df['三個月'].astype(float)
    fund_df['六個月'] = fund_df['六個月'].astype(float)
    fund_df['一年'] = fund_df['一年'].astype(float)
    fund_df['二年'] = fund_df['二年'].astype(float)
    fund_df['三年'] = fund_df['三年'].astype(float)
    fund_df['五年'] = fund_df['五年'].astype(float)
    fund_df['自今年以來'] = fund_df['自今年以來'].astype(float)

    half_of_row_count = len(fund_df.index) // 2 # 計算DataFrame行數的一半，將用於選擇前50%的數據

    rule_3_df = fund_df.sort_values(by=['三年'], ascending=['True']).nlargest(half_of_row_count, '三年') # 根據 "三年" 列的值對 fund_df 進行升序排序，然後選擇 "三年" 列值最大的前 half_of_row_count 行
    # fund_df.sort_values(by=['三年'], ascending=['True'])：這段代碼對 fund_df 進行排序。
    # by=['三年'] 表示根據 "三年" 列的值進行排序
    # ascending=['True'] 表示按升序排序
    # .nlargest(half_of_row_count, '三年')：這段代碼從排序後的DataFrame中選擇 "三年" 列值最大的前 half_of_row_count 行

    
    rule_1_df = fund_df.sort_values(by=['一年'], ascending=['True']).nlargest(half_of_row_count, '一年') # 根據 "一年" 列的值對 fund_df 進行升序排序，然後選擇 "一年" 列值最大的前 half_of_row_count 行
    rule_6_df = fund_df.sort_values(by=['六個月'], ascending=['True']).nlargest(half_of_row_count, '六個月') # 根據 "六個月 " 列的值對 fund_df 進行升序排序，然後選擇 "一年" 列值最大的前 half_of_row_count 行

    rule_31_df = pd.merge(rule_3_df, rule_1_df, how='inner') # 將 rule_3_df 和 rule_1_df 進行內連接 (inner join)，僅保留同時存在於兩個DataFrame中的行。
    # 內連接: 只保留在兩個表格中都存在的那些行。換句話說，內連接只保留在兩個表格中都有匹配鍵值的行 (交集)
    rule_316_df = pd.merge(rule_31_df, rule_6_df, how='inner')
    # pd.merge(rule_3_df, rule_1_df, how='inner')：pd.merge 是 pandas 的合併函數，how='inner' 指定了連接方式為內連接。

    fund_rule_items_str = '' # 用於存儲最終的基金信息

    if not rule_6_df.empty: # 檢查 rule_6_df 是否為空。如果不為空，則執行接下來的操作
        for _, row in rule_316_df.iterrows(): # iterrows() 是 pandas 提供的一個方法，用於生成一個迭代器，這個迭代器返回DataFrame中的每一行的索引和行數據
            fund_rule_items_str += f'{row["基金名稱"]}, {row["三年"]}, {row["一年"]}, {row["六個月"]}\n'

    return fund_rule_items_str


@app.route("/", methods=['GET'])
def hello():
    return 'hello heroku'


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_input = event.message.text
    if user_input == '@基金列表':
        fund_list_str = ''
        for fund_name in fund_map_dict:
            fund_list_str += fund_name + '\n'
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=fund_list_str))
    elif user_input in fund_map_dict:
        group_id = fund_map_dict[user_input]
        print('開始篩選...')
        fund_rule_items_str = fetch_fund_rule_items('2020', '02', group_id)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=fund_rule_items_str))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text='請輸入正確指令'))


init_fund_list()

"""
init_fund_list() 需要移出 if 條件判斷外（這部份課程內容有誤植，若是在本機端沒問題，但部屬到 heroku 上若放在裡面就不回執行到）
__name__ 為 Python 內建變數，若程式不是被當作模組引入則 __name__ 內容為 __main__ 字串，直接在本機電腦端執行此檔案會進入執行 app.run()
但在 heroku 上會被 gunicorn 這個伺服器當作模組使用所以 name 為此檔案名稱 ex. line_app，不會進入執行，我們透過 Procfile 設定執行我們伺服器。
"""
if __name__ == '__main__':
    app.run()