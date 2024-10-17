'''
1. 目標: 
    1. 整合 Google Sheets API 寫入試算表資料
    2. 整合 Google Sheets API 讀取試算表資料
    3. 根據估價公式推送預警通知


2. Google Sheets: 
    Google 雲端服務的其中一個重要產品。
    它可以讓你透過瀏覽器介面操作試算表並分享給其他使用者，雖然功能不一定比 Excel 完整，
    但雲端服務有更彈性的使用方式。只要我們有 Google 帳號就能免費在一定的限制下使用
        1. Spreadsheet (試算表): 
            Google Sheets 試算表的基本單位，當我們新增檔案時就是新增一個試算表

        2. Worksheet (工作表): 
            為試算表下的分頁單位，每個試算表可以有多個工作表。為切分不同的工作資料的單位

        3. Column (欄):
            在工作表中直排資料

        4. Row (列): 
            在工作表中橫排資料

        5. Cell (儲存格):   
            指定的欄和列的資料區域，也是我們輸入資料值的地方


3. Python Google Sheets API : 
    接下來將我們的資料儲存到 Google Sheets 上 (當然也可以使用其他儲存方式，例如: 檔案或資料庫)。
    方便我們接下來 LINE Bot API 查詢和比對使用。

        1. 設定 Google Sheets API:
            我們需要進入 Google Sheets API 後點選【新增專案】啟用 Google Sheets API
        
        2. 申請 Service Account 服務帳號:
            Service Account 是一種專為應用程式而設的 Google 帳號。
            與個人 Google 帳號不同，Service Account 不對應任何特定的使用者。
            而是對應一個應用程式或一個虛擬的實體，這使得應用程式可以以 Service Account 的身份與 Google API 互動。

        3. 金鑰(Key):
            金鑰是一個用於身份驗證的數字證書，它由【公鑰】和【私鑰】組成。
            這些密鑰用於加密和解密信息，確保通信的安全性和身份的真實性。

                1. 私鑰(Private Key):
                    私鑰是一個用於加密的秘密數字證書。
                    只有擁有這個私鑰的應用程式或服務才能解密由相應公鑰加密的信息。
                    這確保了只有被授權的應用程式才能訪問敏感數據。

                    創建 Service Account 並生成金鑰時， Google 會提供一個 JSON 文件，
                    這個文件包含私鑰。應用程式需要使用這個私鑰來進行身份驗證
                    

                2. 公鑰(Pubkic Key):
                    公鑰是公開的，可以用來加密信息。只有對應的私鑰才能解密這些信息。
                    公鑰通常存儲在 Google 的服務器上，當你的應用程式使用私鑰進行身份驗證時，
                    Google 會使用公鑰來驗證這個過程。


4. 安裝串接 Google API 套件:
    接著在終端機安裝操作試算表的套件。
    使用 gspread 這個第三方套件 (包裝方便使用) 和驗證服務帳號套件 oauth2client
    進行 Google Sheets API 的操作。

        1. gspread:
            用於操作 Google Sheets 的第三方 Python 套件。
            它提供了一個簡單的接口，讓你可以輕鬆地讀取和寫入 Google Sheets 的數據。

        2. oauth2client:
            用於處理 OAuth 2.0 認證(業界標準協議，用於進行身份驗證和授權) 和授權的 Python 庫。
            它允許你使用服務帳號來進行身份驗證，以便安全地訪問 Google API。
        
                # pip install gspread oauth2client


5. Google sheet 操作:
    1. Google sheet 右上角共用
    2. 將金鑰中的 client_email 欄位貼過來【使用者輸入框】將權限給予服務帳號
    3. 把 Google Sheets 網址中試算表 ID 複製下來 (白色區塊，不含前後的 /)，同樣輸出到環境變數上
        環境變數為系統使用的變數，主要用來儲存參數設定值，讓系統中的程式可以讀取使用
            
            格式: https://docs.google.com/spreadsheets/d/{複製這個 Key}/edit#gid=0


6. 終端機操作: 

    set SPREAD_SHEETS_KEY=xxxxxxx
    
    # 列印出 %SPREAD_SHEETS_KEY% 看是否有設置成功
    echo %SPREAD_SHEETS_KEY%


7. 測試金鑰是否可以正常使用:
    在 Python 使用 Google Sheets API 金鑰有兩種方式。選擇一種使用即可：
        1. 直接讀取金鑰檔案
        2. 使用環境變數讀取金鑰資訊
'''

# 開始操作

'''
1. 直接讀取金鑰檔案

    1. 將金鑰檔案和程式放在同一個資料夾。
    2. 使用 ServiceAccountCredentials.from_json_keyfile_name 方法讀取金鑰檔案

    
1.1 code邏輯:

    1. 身份驗證:
        使用服務帳號的金鑰文件進行身份驗證，證明我們的應用程式有權訪問 Google Sheets API

    2. 授權:
        授權範圍(scopes)定義了我們的應用程式對 Google Sheets 的訪問權限 
            授權範圍(scopes): 指定應用程式在訪問用戶資源時的權限範圍。(例如讀取或寫入數據的權限)

    3. gspread 客戶端:
        授權的 gspread 客戶端是指經過【身份驗證】並具有適當授權範圍的 gspread 客戶端對象。
        這個對象允許我們通過程式碼對 Google Sheets 進行操作

'''
import gspread # 用於操作 Google Sheets
from oauth2client.service_account import ServiceAccountCredentials # 用於處理服務帳號的 OAuth 2.0 認證
# 金鑰檔案路徑
credential_file_path = r'C:\Users\user\Desktop\Python\證券投資分析&股票聊天機器人\證券投資分析&股票聊天機器人 Project\股票投資聊天機器人 Project\credentials.json'

# 需要定義應用程式的授權範圍。這裡使用的是 Google Sheets API 的範圍
gsp_scopes = ['https://spreadsheets.google.com/feeds']


# 定義了一個函數 auth_gsp_client，用來讀取金鑰檔案並返回授權的 gspread 客戶端
def auth_gsp_client(file_path, scopes):
    # auth_gsp_client 函數接收兩個參數：file_path (金鑰檔案路徑) 和 scopes (API 授權範圍)

    # 使用 ServiceAccountCredentials.from_json_keyfile_name() 方法從 JSON 金鑰檔案中讀取服務帳號憑證
    credentials = ServiceAccountCredentials.from_json_keyfile_name(file_path, scopes)

    # 使用 gspread.authorize() 方法進行授權並返回一個 gspread 客戶端對象
    return gspread.authorize(credentials)

# 調用 auth_gsp_client 函數並傳入金鑰檔案路徑和 API 授權範圍，來獲取授權的 gspread 客戶端對象
gsp_client = auth_gsp_client(credential_file_path, gsp_scopes)



'''
2. 使用環境變數讀取金鑰資訊

    1. 在終端機使用環境變數儲存金鑰相關資訊。
        這樣就不需要將金鑰檔案提交到 git 上 (資訊安全佳)。但使用步驟較為繁瑣。

    2. 主要操作方式是將【金鑰內容】透過【環境變數】組裝成一個 dict 後。
        傳給 ServiceAccountCredentials.from_json_keyfile_dict 使用。

        
2.1 code邏輯:

    1. 通過設置環境變數來存儲金鑰資訊，避免將敏感資訊直接寫入代碼或提交到版本控制系統。
    
    2. 從環境變數中讀取金鑰資訊，組裝成字典並進行認證，獲取授權的 gspread 客戶端，從而可以安全地操作 Google Sheets。

'''
import os
def get_google_sheets_creds_dict(): # 從環境變數中讀取所有需要的金鑰資訊並組裝成一個字典
    google_sheets_creds = {   
        'type': os.environ.get('GOOGLE_SHEETS_TYPE'),   # os.environ.get 用於獲取環境變數的值
        'project_id': os.environ.get('GOOGLE_SHEETS_PROJECT_ID'),
        'private_key_id': os.environ.get('GOOGLE_SHEETS_PRIVATE_KEY_ID'),
        'private_key': os.environ.get('GOOGLE_SHEETS_PRIVATE_KEY'),
        'client_email': os.environ.get('GOOGLE_SHEETS_CLIENT_EMAIL'),
        'client_id': os.environ.get('GOOGLE_SHEETS_CLIENT_ID'),
        'auth_uri': os.environ.get('GOOGLE_SHEETS_AUTH_URI'),
        'token_uri': os.environ.get('GOOGLE_SHEETS_TOKEN_URI'),
        'auth_provider_x509_cert_url': os.environ.get('GOOGLE_SHEETS_AUTH_PROVIDER_X509_CERT_URL'),
        'client_x509_cert_url': os.environ.get('GOOGLE_SHEETS_CLIENT_X509_CERT_URL')
    }
    return google_sheets_creds

# 取得組成 Google Sheets 金鑰
google_sheets_creds_dict = get_google_sheets_creds_dict()

# auth_gsp_client 是接收金鑰資訊字典和授權範圍作為參數
def auth_gsp_client(creds_dict, scopes): 
    # 在設定環境變數時會系統會將 \n 換行符號轉成 \\n，所以讀入時要把它替換回來
    creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')

    # 使用 ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes) 來創建憑證對象
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes) 
    
    # 使用 gspread.authorize(credentials) 返回授權的 gspread 客戶端對象。
    return gspread.authorize(credentials)

gsp_client = auth_gsp_client(google_sheets_creds_dict, gsp_scopes)

