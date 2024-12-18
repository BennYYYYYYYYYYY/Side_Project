'''
1. 外幣管理聊天機器人專案主要功能:

    1. 讓使用者可以查詢匯率資訊
    2. 讓使用者可以紀錄外匯交易資料、查詢目前損益 

2. 認識外幣匯率

    1. 現金匯率
        1. 交易過程出現現鈔，【買賣外幣現鈔】

    2. 即期匯率
        1. 交易過程沒有出現現鈔，【帳戶中買賣外幣】

'''
# twder 匯率查詢套件
import twder

twder.currencies() # 取得幣別清單列表

twder.now_all() # 取得目前所有幣別匯率
# # {貨幣代碼: (時間, 現金買入, 現金賣出, 即期買入, 即期賣出), ...}

twder.now('JPY') # 取得目前特定幣別匯率報價
# # (時間, 現金買入, 現金賣出, 即期買入, 即期賣出)

twder.past_day('JPY') # 取得昨天特定幣別匯率報價
# [(時間, 現金買入, 現金賣出, 即期買入, 即期賣出), ...]

twder.past_six_month('JPY') # 取得過去半年特定幣別匯率報價
# [(時間, 現金買入, 現金賣出, 即期買入, 即期賣出), ...]

twder.specify_month('JPY', 2019, 12) # 取得特定期間月份特定幣別匯率 2019(年), 12(月)
# [(時間, 現金買入, 現金賣出, 即期買入, 即期賣出), ...]

