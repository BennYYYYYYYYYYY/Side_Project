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
from apscheduler.schedulers.blocking import BlockingScheduler


sched = BlockingScheduler()
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

    print(f'便宜價:{cash*15}\n合理價:{cash*20}\n昂貴價:{cash*30}')

#    with open('stock_price.csv', 'w', encoding = 'utf-8-sig') as file:
#        file.write(f'便宜價,合理價,昂貴價\n')
#        file.write(f'{cash*15},{cash*20},{cash*30}\n')

# decorator 設定 Scheduler 的類型和參數，例如 interval 間隔多久執行
@sched.scheduled_job('interval', minutes=5)
def timed_job():
    # 要注意不要太頻繁抓取，但測試時可以調整時間少一點方便測試
    print('每 5 分鐘執行一次程式工作區塊')
    get_stock_price()

# 開始執行
sched.start()
