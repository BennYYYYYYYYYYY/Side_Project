'''
在本機端測試時可以加入本機電腦的環境變數，格式為由一個 key 對應一個 value 值 (key 為自行命名)
(注意若要部屬到 Heroku 上要另外在 Heroku 後台加入環境變數，本機電腦端主要為開發測試使用)
'''

'''
1. 主動傳送訊息    
    若是需要主動傳送資訊給使用者需要使用Push API【push_message】而非 reply_message 。
    然而 Push API 訊息在免費額度下有每月 500 則的限制，所以要留意不過度使用。
        
        使用方式，需填入 USER_ID，messages 可以攜帶多個訊息元件：

            line_bot_api.push_message(
                to={USER_ID},
                messages=[
                    TextSendMessage(text='主動推送的文字')
                ]
            )


2. 把 user-ID 存入環境變數中
    可以從 LINE App 後台查詢到自己的 User ID (Basic Settings)

            set LINE_USER_ID=xxxxxxx
            echo %LINE_USER_ID%              # 列印出 %LINE_USER_ID% 看是否有設置成功


3. 設定 LINE secret/token
    需要將 LINE_CHANNEL_ACCESS_TOKEN 和 LINE_CHANNEL_SECRET 加入環境變數測試

        set LINE_CHANNEL_ACCESS_TOKEN=xxxxxxx
        set LINE_CHANNEL_SECRET=xxxxxxx

        
4. 這樣一來 Python 程式碼就可以從環境變數讀到需要的憑證參數【可以print(LINE_CHANNEL_ACCESS_TOKEN)檢查】
    
    # 從環境變數取出設定參數
    LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
    LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

        此時在同一個終端機執行啟動 python XXXXX.py 可以成功啟動本機伺服器



'''

