'''
使用 【connection】 = sqlite3.connect() 連接資料庫，建立連線
使用 【connection】.cursor() 建立 【cursor】 資料庫指標物件
使用 【cursor】.execute() 執行資料庫 SQL 操作
呼叫 【connection】.close() 關閉資料庫連線

【connection】: 操作連接資料庫
【cursor】: 操作資料庫內資料
'''
import sqlite3

# 建立資料庫連線 (若無資料庫則新建一個 SQLite 資料庫，本身是一個檔案) 以下會建立 demo.db 資料庫
connection = sqlite3.connect('demo.db') 

# 使用資料庫 cursor 指標進行 SQL 操作
cursor = connection.cursor()
'''
執行 SQL 語法，新增 stocks 資料表，欄位包含:

    1. id (整數 INT) 【為識別資料的主要欄位(PRIMARY KEY)】
    2. company_name (字串 TEXT)
    3. price (整數 INT)

NOT NULL 代表欄位不得為空值
'''
cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS stocks (
                id INT PRIMARY KEY NOT NULL,
                company_name TEXT NOT NULL,
                price INT NOT NULL
        );
        '''
)

# commit 代表提交，才是真正的將指令在資料庫執行
connection.commit()
# 最後操作完指令記得關閉資料庫連線
connection.close()

# =================================================================================================

# 使用資料庫 cursor 指標進行 SQL 操作
cursor = connection.cursor()

# 新增一筆股票資料
cursor.execute(
        '''
        INSERT INTO stocks (id, company_name, price)
        VALUES (2330, '台積電', 220);
        '''
)

connection.commit()

# 新增一筆資料
cursor.execute(
        '''
        INSERT INTO stocks (id, company_name, price)
        VALUES (2317, '鴻海', 82);
        '''
)

connection.commit() # 儲存, 保存至database

# 最後操作完指令記得關閉資料庫連線
connection.close()

# =================================================================================================

# 使用資料庫 cursor 指標進行 SQL 操作
cursor = connection.cursor() 

# 查詢資料，* 代表所有欄位
# cursor.execute 返回的是一個包含查詢結果的迭代器或游標物件 (這邊返回我select的內容)
rows = cursor.execute(
        '''
        SELECT * 
        FROM stocks;
        '''
)

# 將查詢資料使用 for 迴圈印出，rows[0] 代表第一個欄位，依此類推
for row in rows:
        print(f'id:{row[0]}, company_name:{row[1]}, price:{row[2]}')


# 最後操作完指令記得關閉資料庫連線
connection.close()

# =================================================================================================

cursor = connection.cursor()

cursor.execute(
    '''
    UPDATE stocks
    SET company_name = 'TSMC'
    WHERE id = 2330;
    '''
)   

connection.commit()

rows = cursor.execute(
    '''
    SELECT *
    FORM stocks;
    '''
)

for row in rows:
        print(f'id:{row[0]}, company_name:{row[1]}, price:{row[2]}')

connection.close()

# =================================================================================================

cursor = connection.cursor()

cursor.execute(
    '''
    DELETE FROM stocks
    WHERE id = 2330;
    '''
)

connection.commit()

connection.close()

# ============================================================================================

import pandas as pd

df = pd.read_csv(r'C:\Users\user\Desktop\Python\SQLite\stocks.csv', encoding='utf-8')
stock_data = df.loc[:, ['證券代號', '證券名稱', '收盤價']]

with sqlite3.connect('stock_db.db') as connection:
    cursor = connection.cursor()
    

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stocks (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            closing_price INT NOT NULL
        );
        """
    )
    connection.commit()

    # 插入資料
    for index, row in stock_data.iterrows():
        cursor.execute(
            '''
            INSERT INTO stocks (id, name, closing_price)
            VALUES (?, ?, ?);
            ''', 
            (f'00{row[0]}', row[1], row[2])
        )
    connection.commit()

    # 查詢資料
    rows = cursor.execute(
        '''
        SELECT * 
        FROM stocks WHERE closing_price > 30
        '''
    )

    # 列印結果
    for row in rows:
        print(f'id: {row[0]}, name: {row[1]}, closing_price: {row[2]}')


