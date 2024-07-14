import os

import requests
import pandas as pd
from bs4 import BeautifulSoup

from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)

from linebot.exceptions import (
    InvalidSignatureError
)

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = Flask(__name__)


headers = {
    'user-agent': ''
}


LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

fund_map_dict = {}


def init_fund_list():
    """
    初始化建立基金列表（也可以使用 Google Sheets 儲存，這邊我們儲存在 dict 中）
    """
    resp = requests.get('https://www.sitca.org.tw/ROC/Industry/IN2421.aspx?txtMonth=02&txtYear=2020', headers=headers)
    soup = BeautifulSoup(resp.text, 'lxml')

    table_content = soup.select('#ctl00_ContentPlaceHolder1_TableClassList')[0] # id = 'ctl00_ContentPlaceHolder1_TableClassList'

    fund_links = table_content.select('a') # <a>標籤

    for fund_link in fund_links:
        if fund_link.text:
            fund_name = fund_link.text
            fund_group_id = fund_link['href'].split('txtGROUPID=')[1]
            fund_map_dict[fund_name] = fund_group_id


def fetch_fund_rule_items(year, month, group_id):
    fetch_url = f'https://www.sitca.org.tw/ROC/Industry/IN2422.aspx?txtYEAR={year}&txtMONTH={month}&txtGROUPID={group_id}'
    print(year, month, group_id, fetch_url)
    resp = requests.get('https://www.sitca.org.tw/ROC/Industry/IN2422.aspx?txtYEAR=2020&txtMONTH=02&txtGROUPID=EUCA000523', headers=headers)

    soup = BeautifulSoup(resp.text, 'lxml')

    table_content = soup.select('#ctl00_ContentPlaceHolder1_TableClassList')[0]


    fund_df = pd.read_html(table_content.prettify(), encoding='utf-8')[1] 

    fund_df = fund_df.drop(index=[0]) # index=[0]: 指定要刪除的行的索引 (這裡是第0 row)
    fund_df.columns = fund_df.iloc[0] # iloc: 用於基於整數位置 (從0開始) 選擇row數據
    fund_df = fund_df.drop(index=[1])
    fund_df.reset_index(drop=True, inplace=True)
    fund_df = fund_df.fillna(value=0) # fillna: 用於填充缺失值。value=0: 所有的缺失值都被替換為0

    fund_df['一個月'] = fund_df['一個月'].astype(float)
    fund_df['三個月'] = fund_df['三個月'].astype(float)
    fund_df['六個月'] = fund_df['六個月'].astype(float)
    fund_df['一年'] = fund_df['一年'].astype(float)
    fund_df['二年'] = fund_df['二年'].astype(float)
    fund_df['三年'] = fund_df['三年'].astype(float)
    fund_df['五年'] = fund_df['五年'].astype(float)
    fund_df['自今年以來'] = fund_df['自今年以來'].astype(float)

    half_of_row_count = len(fund_df.index) // 2

    rule_3_df = fund_df.sort_values(by=['三年'], ascending=['True']).nlargest(half_of_row_count, '三年')
    # 這一步是對 fund_df 進行排序。
    # by=['三年'] 表示按名為 "三年" 的列進行排序。
    # ascending=['True'] 表示以升序排序。
    # 在排序完成後，這一步選取 "三年" 列值最大的前 half_of_row_count 行

    rule_1_df = fund_df.sort_values(by=['一年'], ascending=['True']).nlargest(half_of_row_count, '一年')
    rule_6_df = fund_df.sort_values(by=['六個月'], ascending=['True']).nlargest(half_of_row_count, '六個月')

    rule_31_df = pd.merge(rule_3_df, rule_1_df, how='inner')
    rule_316_df = pd.merge(rule_31_df, rule_6_df, how='inner')

    fund_rule_items_str = ''

    if not rule_6_df.empty:
        for _, row in rule_316_df.iterrows():
            # iterrows() 是 pandas 提供的一個方法，用於生成一個迭代器，這個迭代器返回 DataFrame 中每一行的索引和行數據
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