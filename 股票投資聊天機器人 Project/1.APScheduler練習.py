'''
功能規劃
    1. 定期執行程式抓取指定個股財務資料並根據估價公式計算出個股便宜價、合理價和昂貴價更新到 Google Spread Sheet 試算表上
    2. 定期讀取 Google Spread Sheet 試算表財務資料，判斷目前股價是否到合適買賣點並發送通知給使用者
'''


'''
1. 定期執行: 
    顧名思義就是在固定一段時間於背景自動執行的程式 (例如：定期更新)

2. 定期執行程式概念:
    1. 定期在背景執行程式是許多網路服務常見的功能之一 (通常稱為 Scheduler)
    2. 在一般伺服器中可以使用 cron 這個工具去定期執行我們的工作程式 (job)

3. 如何在 Python 定期執行程式:
    在 Python 中有一個輕量的 APScheduler 套件。
    可以讓我們使用 Python 程式撰寫我們固定要執行的程式

4. 安裝 APScheduler 套件
    # pip install apscheduler

'''
# BlockingScheduler 是一個阻塞型的排程器，它會一直運行直到被中斷
from apscheduler.schedulers.blocking import BlockingScheduler

# 創建一個 Scheduler 物件實例，這個實例將用來添加和管理定時任務
sched = BlockingScheduler()

# sched.scheduled_job 是 apscheduler 提供的一個方法，這個方法返回一個裝飾器
# decorator 設定 Scheduler 的類型和參數，例如 interval 間隔多久執行
@sched.scheduled_job('interval', seconds=1) # 這個裝飾器告訴 sched，我們要創建一個間隔執行的任務
# 參數 'interval' 表示這是一個間隔任務，seconds=1 表示每隔一秒鐘執行一次
def timed_job():
    print('每 1 秒鐘執行一次程式工作區塊')

# decorator 設定 Scheduler 為 cron 固定每週週間 6pm. 執行此程式工作區塊
@sched.scheduled_job('cron', day_of_week='mon-fri', hour=18) # 這個裝飾器告訴 sched，我們要創建一個固定時間的任務
# 參數 'cron' 表示這是一個基於時間的排程任務
# day_of_week='mon-fri' 表示這個任務在每週一到週五執行
# hour=18 表示在每天下午 6 點 (UTC 時間) 執行
def scheduled_job():
    print('每週週間 6 PM UTC+0，2 AM UTC+8. 執行此程式工作區塊')

# 開始執行
sched.start()