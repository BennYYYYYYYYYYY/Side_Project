# 查詢即時股價
'''
twstock: 專為台灣股市設計的 Python 第三方套件。
    
    它提供了簡單易用的接口，讓開發者能夠方便地擷取台灣股票的即時價格、歷史價格以及相關的股票資訊。
    twstock 使用台灣證券交易所 (TWSE) 和櫃買中心 (TPEx) 提供的公開 API 來獲取數據。
        # pip install twstock 
        # pip install lxml
'''
import twstock
realtime_data = twstock.realtime.get('2330')

print(realtime_data)

'''        
    1. 擷取即時股價
        可以用來獲取當前的即時股價。
'''
 # 擷取 2330 的即時數據，並將結果存儲在 realtime_data 變數中
# stock_name = realtime_data['info']['name'] # 從 realtime_data 字典中的 'info' 鍵下提取股票名稱，存儲在 stock_name 變數中
print(realtime_data.values())

{
    'success': True,
    'info': {
        'time': '2021-12-31 14:30:00', # 資料的更新時間
        'code': '2330',  # 股票代碼
        'channel': '2330.tw',   # 股票的頻道標識
        'name': '台積電',    # 股票的簡稱
        'fullname': '台灣積體電路製造股份有限公司',  # 股票的全稱
        'realtime_start': '2021-12-31 14:30:00',  # 即時數據開始時間
        'realtime_end': '2021-12-31 14:30:00',    # 即時數據結束時間
        'provider': 'twstock'  # 資料提供者
    },
    'realtime': {
        'latest_trade_price': '650', #  最近一次交易價格
        'best_bid_price': ['649', '648', '647', '646', '645'], #  最佳買價列表，依序為 5 個最佳【買價】
        'best_bid_volume': ['10', '20', '30', '40', '50'], #  對應最佳買價的【買量】
        'best_ask_price': ['651', '652', '653', '654', '655'], #  最佳賣價列表，依序為 5 個最佳【賣價】
        'best_ask_volume': ['5', '15', '25', '35', '45'], #  對應最佳賣價的【賣量】
        'open': '645', # 當日開盤價
        'high': '655', # 當日最高價
        'low': '640', # 當日最低價
        'accumulate_trade_volume': '10000' # 當日累積交易量
    }
}

'''
    2. 擷取歷史股價
        可以用來獲取指定日期範圍內的歷史股價。
'''
stock = twstock.Stock('2330')  # 2330 是台積電的股票代碼
historical_prices = stock.price  # 最近 30 天的收盤價
print(historical_prices)

'''
    3. 提供基本的股票技術指標
        如移動平均線 (MA)、相對強弱指數 (RSI) 等。
'''
# 參數為最近 30 天的收盤價 stock.price 和天數 5
moving_average_5 = stock.moving_average(stock.price, 5)
print(moving_average_5)

#  rsi 方法計算 14 天的相對強弱指數 (RSI)
rsi_14 = stock.rsi(14)
print(rsi_14)



