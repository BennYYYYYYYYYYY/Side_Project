'''
1. 準備階段
    導入所需的庫:
        1. 操作系統操作 (os)
        2. 處理 JSON 數據(json)
        3. Google Sheets API 客戶端 (gspread)以及
        4. 處理 Google API 認證的庫 (oauth2client.service_account)

        
2. 設定 Google Sheets API 權限範圍
    定義權限範圍:
        設定應用程序請求的權限範圍，這裡使用了 ['https://spreadsheets.google.com/feeds']
        意味著應用程序需要對 Google Sheets 進行讀寫操作的權限

        
3. 讀取環境變數
    讀取 Google Sheets Key:
        從操作系統的環境變數中讀取 Google Sheets 的鍵 (Spreadsheet Key) 【事先要存進環境變數中】
        這個鍵用來標識要操作的特定 Google Sheets。

        
4. 設定認證密鑰文件路徑
    指定密鑰文件路徑:
        設置指向存放 Google API 認證金鑰 JSON 文件的路徑。
        這個文件是從 Google Cloud Platform 上創建服務帳戶後下載的。

        
5. 建立授權客戶端
    1. 定義授權函數：
        創建一個函數來讀取密鑰文件並進行授權。
        使用服務帳戶憑證從 JSON 文件中讀取憑證，並使用這些憑證授權 gspread 客戶端。

    2. 獲取授權的 gspread 客戶端：
        調用這個授權函數，傳入密鑰文件路徑和權限範圍，返回一個已授權的 gspread 客戶端對象。
        

6. 操作 Google Sheets 工作表
    打開 Google Sheets:
        使用授權的 gspread 客戶端對象，根據 Spreadsheet Key 打開 Google Sheets
        並選擇要操作的工作表 (這裡是 sheet1)。

        
7. 清除工作表數據：
    1. 使用 gspread 提供的 clear 方法清除工作表中的所有現有數據
        確保數據操作是從空白狀態開始的。

    2. 插入新數據:
        使用 gspread 的 insert_row 方法將新的數據插入到工作表的第一行
        這裡插入的數據是 ['測試資料欄 1', '測試資料欄 2']。
'''

import os # 用於與操作系統交互，這裡用來讀取環境變數
import json

# 引入套件
import gspread
from oauth2client.service_account import ServiceAccountCredentials 

# 我們使用 Google API 的範圍為 spreadsheets
gsp_scopes = ['https://spreadsheets.google.com/feeds'] # ['https://spreadsheets.google.com/feeds']: 這是一個 URL，代表 Google Sheets API 的讀寫權限


# 使用 os 模組的 environ.get 方法從環境變數中獲取 Google Sheets 的鍵 (Spreadsheet Key)，用於識別要操作的 Google Sheets
SPREAD_SHEETS_KEY = os.environ.get('SPREAD_SHEETS_KEY')

# 金鑰檔案路徑
credential_file_path = r''

# auth_gsp_client 函數用來創建並返回一個經過認證的 gspread 客戶端對象
def auth_gsp_client(file_path, scopes):
    # 從檔案讀取金鑰資料
    
    credentials = ServiceAccountCredentials.from_json_keyfile_name(file_path, scopes) 
    # 使用 ServiceAccountCredentials 類的 from_json_keyfile_name 方法，從指定的 JSON 文件中讀取憑證，並使用這些憑證【創建一個憑證對象】
    
    return gspread.authorize(credentials) # 使用 gspread 庫的 authorize 方法，根據憑證對象進行授權
    # 並返回一個授權的 gspread 客戶端對象

# 使用 auth_gsp_client 函數並傳入金鑰文件路徑和權限範圍，獲得經授權的 gspread 客戶端對象
gsp_client = auth_gsp_client(credential_file_path, gsp_scopes)

# 使用 open_by_key 方法根據 SPREAD_SHEETS_KEY 打開 Google Sheets，並選擇第一個工作表 (sheet1)
worksheet = gsp_client.open_by_key(SPREAD_SHEETS_KEY).sheet1

# 清除工作表中的所有數據
worksheet.clear()

# 使用 insert_row 方法將 ['測試資料欄 1', '測試資料欄 2'] 這兩欄數據插入到第一列
worksheet.insert_row(['測試資料欄 1', '測試資料欄 2'], 1)

'''
1. 憑證 (Credentials)
    1. 憑證是用來驗證應用程序或用戶身份的證明文件。
        它通常包含一組【密鑰】，這些密鑰用來確保應用程序或用戶具有訪問特定資源或服務的權限。

    2. 在這段代碼中的應用:
        1. 服務帳戶憑證:
            在使用 Google Sheets API 時，應用程序需要證明自己的身份，以便獲得讀寫 Google Sheets 的權限。
            
        2. 這種身份驗證通過服務帳戶憑證來完成。
            服務帳戶憑證是一個 JSON 文件，其中包含了一些關鍵信息，
            如 client_email、private_key 等，這些信息用來證明應用程序的身份。

        3. 從 JSON 文件中讀取憑證:
            使用 ServiceAccountCredentials.from_json_keyfile_name(file_path, scopes) 方法從指定路徑的 JSON 文件中讀取憑證。
            這個方法會【加載文件中】的【服務帳戶憑證信息】，並將其【###轉換為一個憑證對象###】。

            
2. 授權 (Authorization)
    1. 授權是指在身份驗證(驗證憑證)的基礎上，給應用程序或用戶【分配特定的權限】
        以便他們能夠執行某些操作或訪問特定資源。

    2. 在這段代碼中的應用:
        1. 範圍定義:
            授權過程中需要定義【應用程序請求的API範圍】(scopes)，這些範圍決定了應用程序可以執行哪些操作。
            在這段代碼中，範圍被設置為 ['https://spreadsheets.google.com/feeds']，表示應用程序請求對 Google Sheets 的讀寫權限。

        2. 授權 gspread 客戶端:
            使用 gspread.authorize(credentials) 方法進行授權，這個方法會使用【前面讀取的憑證】來獲取一個【###授權的 gspread 客戶端對象###】。
            這個對象擁有了對 Google Sheets API 執行讀寫操作的權限。
'''