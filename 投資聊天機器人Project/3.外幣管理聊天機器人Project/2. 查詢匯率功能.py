'''
首先先建立一個函式，當呼叫這個函式時我們會將 twder 套件查詢到的所有幣別匯率資訊組成字串回傳。
由於查詢到的匯率資訊是由一個 dict 所組成（{str: tuple}）。
所以需要將其格式化後整理成易於閱讀的字串後回傳。
'''
import twder

def get_all_currencies_rates_str():
    all_currencies_rates_str = ''
    all_currencies_rates = twder.now_all() # 此時會回傳dict

    # 取出 key, value : {貨幣代碼: (時間, 現金買入, 現金賣出, 即期買入, 即期賣出), ...}
    # (時間, 現金買入, 現金賣出, 即期買入, 即期賣出) 是個 tuple，我們使用 index 取得內含元素
    for currency_code, currency_rates in all_currencies_rates.items():
        all_currencies_rates_str += f'[{currency_code}] 現金買入:{currency_rates[1]} \  # \ 為多行斷行符號（但程式視為同一行），避免程式人閱讀起來過長
            現金賣出:{currency_rates[2]} 即期買入:{currency_rates[3]} 即期賣出:{currency_rates[4]} ({currency_rates[0]})\n'
    return all_currencies_rates_str
