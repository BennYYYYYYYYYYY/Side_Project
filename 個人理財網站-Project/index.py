# 主要伺服器的文件

from flask import Flask, render_template, request, g, redirect
import sqlite3
import requests
import math
import matplotlib.pyplot as plt
import matplotlib
import os

# 用於設定 Matplotlib 後端為 Agg (Anti-Grain Geometry)。這個後端專門用於生成圖像文件，而不會顯示圖形視窗。
matplotlib.use("agg")


app = Flask(__name__) # 讓 Flask 知道目前根目錄是在哪 (1.index.py)

database = "datafile.db"

# 操作資料庫連線
def get_db():
    if not hasattr(g, 'sqlite_db'): # 如果 g object 沒有 has attribute "sqlite_db" 
        g.sqlite_db = sqlite3.connect(database)
    return g.sqlite_db

# teardown_appcontext: 每當處理完一份 http request 後會執行一次
@app.teardown_appcontext
def close_connection(exception):
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()



# 主頁面: 需要顯示所有資料的情況
@app.route("/") 
def home(): 
    # 需要顯示出資料庫的狀況
    conn = get_db()
    cursor = conn.cursor()
    result = cursor.execute("select * from cash") # return a sqlite3 object
    data_result = result.fetchall() # list of tuples，每個 tuple 代表提交的一筆資料(每份表單)
   
    # 計算台幣與美金的總額 (from 資料庫 cash 資料集)
    taiwanese_dollars = 0
    us_dollars = 0
    for data in data_result: # iterate 每份表單
        taiwanese_dollars += data[1] 
        us_dollars += data[2]
        
    # 使用全球匯率API獲取匯率資訊，才能算出總值(台幣計)
    r = requests.get('https://tw.rter.info/capi.php')
    currency = r.json() # currency 是一個 dic 包含所有匯率
    total = math.floor(taiwanese_dollars + us_dollars * currency["USDTWD"]["Exrate"])

    # 取得所有股票資訊 (from 資料庫 stock 資料集)
    result2 = cursor.execute("""select * from stock""")
    stock_result = result2.fetchall()
    # 由於 stock_id 會分開成不同筆資料，所以要把他特別抓出來，unique list
    unique_list = []
    for data in stock_result:
        if data[1] not in unique_list:
            unique_list.append(data[1]) # 讓 list 中的 stock_id 都是唯一的
        
    # 計算股票總市值
    total_stock_value = 0

    # 計算單一股票資訊
    stock_info = []
    for stock in unique_list: # 把每一個 stock_id 的資料抓出來看
        result = cursor.execute("select * from stock where stock_id=?", (stock, )) # (stock, ) 才會是 tuple
        result = result.fetchall()
        stock_cost = 0 # 單一股票總花費
        shares = 0 # 單一股票股數
        for d in result: # result 也是一個 list 包含每一筆表單的資料們
            shares += d[2]
            stock_cost += d[2] * d[3] + d[4] + d[5]
        
        # 取得目前股價
        url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&stockNo=" + stock
        res = requests.get(url)
        data = res.json()
        price_array = data["data"]
        current_price_with_comma = price_array[len(price_array) - 1][6] # 抓當天日期前一天的收盤價
        current_price = float(current_price_with_comma.replace(',', '')) # 把 , 去掉才能轉 float
        
        # 計算單一股票總市值
        total_value = round(current_price * shares)
        total_stock_value += total_value

        # 單一股票平均成本
        average_cost = round(stock_cost / shares, 2)

        # 單一股票報酬率
        rate_of_return = round((total_value - stock_cost) *100 / stock_cost, 2)

        stock_info.append(
            {
                "stock_id":stock,
                "stock_cost":stock_cost,
                "total_value":total_value,
                "average_cost":average_cost,
                "shares":shares,
                "current_price":current_price,
                "rate_of_return":rate_of_return
            }
        )

    # 計算股票佔資產比例 (需要跳脫 for loop，因為要把每個股價拿來計算)
    for stock in stock_info:
        stock["value_percentage"] = round(
            stock["total_value"] * 100 / total_stock_value, 2)


    # 畫股票圓餅圖
    if len(unique_list) != 0:
        labels = tuple(unique_list)
        sizes = [d['total_value'] for d in stock_info]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(sizes, labels=labels, autopct=None, shadow=None) # 不顯示百分比、不添加陰影效果
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0) # 設定上下左右的邊界，以及子圖之間水平與垂直間距
        plt.savefig("static/piechart.jpg", dpi=200) # 解析度為 200 DPI (dots per inch)
    else: # 如果沒有資料就不應該有顯示圓餅圖 (原本有資料後來刪除)
        try: # 避免本來就沒圖
            os.remove("static/piechart.jpg")
        except:
            pass


    # 畫股票現金圓餅圖
    if us_dollars != 0 or taiwanese_dollars != 0 or total_stock_value != 0:
        labels = ("USD", "TWD", "Stock")
        sizes = [us_dollars * currency["USDTWD"]["Exrate"], taiwanese_dollars, total_stock_value]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(sizes, labels=labels, autopct=None, shadow=None) # 不顯示百分比、不添加陰影效果
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0) # 設定上下左右的邊界，以及子圖之間水平與垂直間距
        plt.savefig("static/piechart2.jpg", dpi=200) 
    else: # 如果沒有資料就不應該有顯示圓餅圖 (原本有資料後來刪除)
        try: # 避免本來就沒圖
            os.remove("static/piechart2.jpg")
        except:
            pass


    # 把所有資料放入 object
    data = {
        "show_pic_1":os.path.exists("static/piechart.jpg"), # 是否顯示圓餅圖
        "show_pic_2":os.path.exists("static/piechart2.jpg"),
        "total":total,
        "currency":currency["USDTWD"]["Exrate"],
        "ud":us_dollars,
        "td":taiwanese_dollars,
        "cash_result":data_result,
        "stock_info":stock_info
    }

    # Jinja2 是 Flask 預設的模板引擎，將後端的資料插入HTML模板，讓網頁動態化。{{  }} 和 {%  %} 這些語法就是 Jinja2 的定義方法。
    # data參數: 讓我可以在 html模板(index.html)中獲取data的值，index.html就可以透過 {{ data }} 去存取這邊定義的資料
    return render_template("index.html", data=data)



# 現金頁面
@app.route("/cash")
def cash_form():
    return render_template("cash.html")



# 處理現金頁面的表單 
@app.route("/cash", methods=["POST"])
def submit_cash(): 
    # 取得提交的資料
    taiwanese_dollars = 0
    us_dollars = 0
    if request.values["taiwanese-dollars"] != "": # request 是用來從前端接收資料的物件，request.value 需要用 key-value pair, 而 key 就是根據 HTML 中設定 <input> 中的 name 
        taiwanese_dollars = request.values["taiwanese-dollars"]
    if request.values["us-dollars"] != "": 
            us_dollars = request.values["us-dollars"]
    note = request.values['note']
    date = request.values['date']

    # 更新資料庫
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(""" 
        insert into cash (taiwanese_dollars, us_dollars, note, date_info) 
                   values (?, ?, ?, ?)""", (taiwanese_dollars, us_dollars, note, date))
    conn.commit()

    # 導回主頁面
    return redirect("/")




# 主頁面: 刪除資料按鈕
@app.route("/cash-delete", methods=["POST"])
def cash_delete():
    # 使用者在前端輸入數據 -> 自動在後端生成 id -> 把後端的id放入前端隱藏欄位 -> 在後端把前端的name用request抓回來 -> 指定刪除資料庫的資料
    transaction_id = request.values["id"] # id 在後端原本為自動生成(primary key)，但現在需要用來傳遞刪除哪一筆資料，所以是對應 hidden input 的 name="id"
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""delete from cash where transaction_id=?""", (transaction_id))
    conn.commit()

    return redirect("/")




# 股票頁面
@app.route("/stock")
def stock_form():
    return render_template("stock.html")



# 處理股票頁面的表單
@app.route("/stock", methods=["POST"])
def submit_stock():
    # 取得前端輸入的資料
    stock_id = request.values["stock-id"]
    stock_num = request.values["stock-num"]
    stock_price = request.values["stock-price"]
    processing_fee = 0
    tax = 0
    if request.values["processing-fee"] != "":
        processing_fee = request.values["processing-fee"]
    if request.values["tax"] != "":
        tax = request.values["tax"]
    date = request.values["date"]

    # 更新數據庫資料
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(""" 
        insert into stock (stock_id, stock_num, stock_price, processing_fee, tax, date_info) 
                   values (?, ?, ?, ?, ?, ?)""", (stock_id, stock_num, stock_price, processing_fee, tax, date))
    conn.commit()

    # 導回主頁面
    return redirect("/")




if __name__ == "__main__":
    app.run(debug=True)
