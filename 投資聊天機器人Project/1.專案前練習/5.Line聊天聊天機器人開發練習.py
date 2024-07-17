'''
一般來說 LINE Bot 有五大基本流程步驟：

1. 申請 LINE Bot 開發者帳號並建立 LINE Bot Provider / Channel 等相關設定
2. 環境設定和安裝 Flask、Requests、Python LINE Messaging API SDK 等套件
3. 建立 Python Flask Web Server 並撰寫 Webhook Server 商業邏輯
4. 使用 ngrok 建立 https 入口，讓外界可以使用我們的 Flask Web Server
5. 在 LINE Bot 管理後台設定 Webhook URL 網址後於 LINE App 上測試使用 (可使用 QR Code 加入好友)

LINE Messaging API SDK for Python：LINE Python 聊天機器人開發套件
Python Flask：Python 輕量網頁伺服器框架
Python requests：Python 發送網路請求套件
ngrok：露出本機電腦 IP 位置給外部人可以使用，本機電腦用於開發測試使用

'''

# pip install flask

from flask import Flask, request # Flask 為應用 flask 的核心類別
# request 是來訪問客戶端發送的請求數據

app = Flask(__name__) # 這行程式碼創建了一個 Flask 應用實例 app，接下來所有與應用相關的操作都會藉由這個實例進行。
# __name__ 變數代表目前運行的 Python 模組名稱。
# 當這個模組是主程序時，__name__ 會是 "__main__"，這對於 Flask 來說是一個方便的方式來確定應用程序的根目錄。


# 路由設置
@app.route('/') # app.route('/') 是 Flask 提供的裝飾器，用來將 URL 路徑 / (也就是根路徑) 映射到某個函式。
# 這意味著當有人訪問 http://localhost:5000/ 時，這個 URL 的請求會由裝飾器下方的函式處理。
# 在本機上運行 Flask 應用時，http://localhost:5000/ 是一個預設的訪問 URL。

# 這個路徑 / 表示網站的根目錄。
# 如果改成 @app.route('/hello')，那麼只有當訪問 http://localhost:5000/hello 時，才會由對應的函式處理請求

def hello():
    # # request 是 Flask 提供的一個全局變數，代表當前的 HTTP 請求
    # request.args 包含 URL 查詢字符串中的所有參數。例如，
    # 對於 URL http://localhost:5000/?name=Jack，request.args 會包含 {'name': 'Jack'}

    # request 可以取出網路請求攜帶的資料。
    # 例如：網址參數，ex. request.args.get 方法用來從查詢字符串中提取參數的值。它有兩個參數：
        # 第一個參數是你想要提取的參數名。
        # 第二個參數是當參數不存在時使用的默認值
    
    name = request.args.get("name", "World") 
    # 如果 URL 是 http://localhost:5000/?name=Jack，那麼 name 的值將是 Jack
    # 如果 URL 是 http://localhost:5000/（沒有 name 參數），那麼 name 的值將是默認值 "World"
    
    return f'Hello, {name}!'
    # 這行程式碼生成一個字符串，包含 name 變數的值。這個字符串會作為 HTTP 響應返回給客戶端(即瀏覽器)
'''
在 Flask 中，request 物件代表當前的 HTTP 請求。
每當客戶端(如瀏覽器)發送請求到伺服器，Flask 會自動創建一個 request 物件來存儲這個請求的所有數據。
這些數據包括 URL、HTTP 方法、標頭信息、表單數據、JSON 數據等。

URL 可以包含查詢字符串 (query string)，它是一組以 ? 開始的參數和值對。
例如，在 URL http://localhost:5000/?name=Jack 中，name=Jack 是查詢字符串的一部分
'''

app.run() # 這行程式碼啟動 Flask 應用的內建開發伺服器，使其開始接收並處理 HTTP 請求
# 默認情況下，Flask 伺服器會在 127.0.0.1:5000 上運行
# 其中 127.0.0.1 是本機地址，意味著伺服器只接受來自本機的請求，5000 是端口號 (port number)

# 如果我們想讓伺服器監聽所有網絡介面並在特定端口運行，可以傳遞參數，例如 app.run(host='0.0.0.0', port=8080)
'''
端口號 (port number)
端口號是一個數字，用來識別計算機上的特定應用程序或服務。
它與 IP 地址結合使用來確保數據能夠正確地傳送到網絡上的特定服務或應用程序。
不同的應用程序或服務通常使用不同的端口號來區分，並且你可以根據需要配置和更改端口號
'''

# =================================================================================================

'''
ngrok 
ngrok 是一個工具，它可以幫助我們把本地電腦上的伺服器暴露給外部網絡。
它可以生成一個公開的網址，這個網址可以被任何地方的人訪問，而這些訪問會被轉發到我們的本地伺服器上。

ngrok 的工作原理

1. 創建隧道：
    當我們在本地運行 ngrok 時，它會在我們的電腦上打開一個端口來接收本地的 HTTP 請求。
    同時，ngrok 會在它的雲端服務器上創建一個公開的 URL（例如 https://abc123.ngrok.io）。
    這個公開的 URL 是 ngrok 提供的，可以被外部訪問。
    
2. 轉發請求：
    外部的請求通過這個公開 URL 發送給 ngrok 的雲端服務器。
    ngrok 的雲端服務器會把這些請求通過隧道轉發到我們本地的伺服器。
    我們的本地伺服器處理請求後，再把回應發回給 ngrok 的雲端服務器，然後由 ngrok 把回應傳回給外部客戶端。


3. 實施步驟

    1. 下載 ngrok
    2. 打開flask.py
    3. 把 Anaconda prompt cd 至放 ngrok 的資料夾中
    4. sign up ngrok 得到 Authtoken
    5. 至 Anaconda prompt 輸入 ngrok config add-authtoken <ngrok帳號的authtoken>
    6. Anaconda prompt 輸入 ngrok.exe http 5000

'''

'''
Forwarding(轉發)
Forwarding 是 ngrok 的核心功能，主要用於將外部網絡請求轉發到我的本地伺服器。

    1. 公開 URL: 
        當你啟動 ngrok (例如 ngrok http 5000)，
        ngrok 會生成一個公開的 URL (如 https://abc123.ngrok.io)，這個 URL 可以被外部訪問。
        
    2. 請求轉發:
        任何通過這個公開 URL 發送的 HTTP 請求，都會被 ngrok 的雲端服務器接收到，
        然後通過一個安全的隧道轉發到你的本地伺服器 (例如 http://localhost:8080)。

    3. 回應轉發:
        本地伺服器處理請求並生成回應，ngrok 會將這個回應通過隧道返回給外部客戶端。
        這個過程確保了你的本地伺服器能夠接收到來自外部的請求，並且可以正常處理和回應這些請求。

        
Web Interface (網頁界面)
ngrok 提供了一個本地的 Web Interface，用於檢視和調試所有通過 ngrok 隧道進入的 HTTP 請求。
這個界面通常運行在 http://127.0.0.1:4040，你可以在瀏覽器中打開這個地址來查看。以下是 Web Interface 的一些功能：

    1. 請求列表:
        你可以看到所有通過 ngrok 隧道進來的 HTTP 請求，包括請求的詳細信息(如路徑、參數、標頭等)。

    2. 回應詳情:
        除了請求，你還可以查看每個請求的回應詳情，包括 HTTP 狀態碼、回應時間和回應的內容。

    3. 重發請求:
        Web Interface 允許重發以前的請求，這對於調試非常有用，
        因為可以在不重新觸發外部事件的情況下反覆測試你的本地伺服器。
    
    4. 請求和回應的內容: 
        可以查看和分析 HTTP 請求和回應的內容，這對於理解和解決問題非常有幫助。
'''

# =================================================================================================

'''
1. Messaging API 是我們要使用來開發聊天機器人的 API。

    Messaging API 是 LINE 提供的一個強大工具，用於開發與 LINE 用戶互動的機器人應用程式。
    透過 Messaging API，開發者可以接收來自用戶的訊息，並且回覆這些訊息，從而實現多種互動功能。

    
2. 主要功能

    1. 接收訊息:
        當用戶向你的 LINE 官方帳號發送訊息時，這些訊息會被轉發到你設定的 Webhook 伺服器。
        Webhook 是一個網絡地址(URL)，用來接收 LINE 伺服器發送的 HTTP POST 請求，其中包含用戶的訊息數據。
    
    2. 回覆訊息:
        你可以根據用戶發送的訊息，通過 LINE 的回覆 API (Reply API) 來回覆訊息。
        回覆訊息通常是基於用戶的某一特定訊息事件觸發的，這樣能確保回覆訊息直接與用戶的訊息對應。
    
    3. 推送訊息:
        除了回覆訊息，還可以主動推送訊息給用戶，這稱為推送 API (Push API)。
        推送訊息不需要等用戶先發送訊息，可以在任何時間點發送給用戶。

    4. 豐富的訊息格式:
        LINE 的 Messaging API 支持多種訊息格式，包括文字訊息、圖片訊息、音訊訊息、貼圖訊息、模板訊息等。
        模板訊息可以包括按鈕、確認、卡片等互動元素，讓訊息更加生動和可交互。

    5. 多種事件處理：
        Messaging API 可以處理多種用戶事件，不僅僅是訊息事件，
        還包括加好友事件、取消好友事件、加入群組事件、離開群組事件等。

        
3. 工作流程

以下是 Messaging API 的基本工作流程：

    1.用戶發送訊息:
        用戶在 LINE 中發送訊息給你的官方帳號。

    2. Webhook 接收訊息:
        LINE 伺服器將用戶的訊息 作為 HTTP POST 請求發送到你的 Webhook URL。
        請求的數據中包含用戶的訊息和事件詳情。
        
    3. 伺服器處理訊息:
        你的 Webhook 伺服器接收到請求後，根據訊息內容和事件類型進行處理。
        你可以使用程式邏輯決定如何回應用戶的訊息，例如回覆一個預設的文字訊息、發送圖片、推送特定通知等。
    
    4. 回覆用戶:
        根據處理結果，你可以使用回覆 API 回覆用戶的訊息，或者使用推送 API 主動推送訊息給用戶。
'''


'''
首先，我們先到管理後台設定 Provider 和 Channel (可以想成一個 Chatbot 就是一個 Channel，
可以自行取名。category and subcategory 分類則是根據你的 Channel 種類自行選擇合適分類）

    1. Create 一個 Provider:
        可以是自行命名的名稱

    2. 在 Provider 下面可以設定 Messaging API Channel

    3. 由於一開始預設是會回傳預設的回覆:
        但我們要啟用 Webhook 並關閉 Auto-reply messages 讓我們可以透過程式根據使用者的輸入來處理不同情境。

    4. 設定 LINE Webhook URL: 
        登錄到你的 LINE 官方帳號管理後台，將 ngrok 提供的公開 URL 配置為 Webhook URL (Forwarding 行中顯示的 URL)
        
'''

# =================================================================================================

'''
1. SDK (Software Development Kit, 軟體開發套件)

    SDK 是指軟體開發套件，是一組工具、程式庫、相關文件和範例程式碼的集合，提供給開發者使用，
    幫助他們更高效地開發應用程式。這些工具和程式庫通常是針對某一特定平台、框架或 API 的，
    以便開發者能夠輕鬆地與該平台或服務進行整合。

    
2. 完整的 SDK 通常包含：

    1. API (Application Programming Interface):
        一組定義和規範，說明如何與特定服務、應用或平台進行交互。API 通常是 SDK 的核心部分。
    
    2. 程式庫和框架:
        SDK 提供的程式庫或框架，包含常用的功能和工具，可以幫助開發者快速開發應用程式。
        例如，Python SDK 可能包含多個 Python 模組，這些模組封裝了與特定服務交互的邏輯。

    3. 文件和範例程式碼:
        詳細的開發文檔和範例程式碼，幫助開發者了解如何使用 SDK 提供的功能和 API。
        這些文件通常包含 API 參考、使用指南和最佳實踐。
    
    4. 開發工具:
        一些 SDK 可能還包含圖形化的開發工具或命令行工具，幫助開發者更方便地進行開發、測試和調試工作。

'''

# pip install line-bot-sdk
# pip install flask
# pip install requests

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from flask import Flask, request, abort
# Flask 是一個輕量級的 Web 框架，用於建立 Web 應用程序。
# request 用於處理 HTTP 請求。
# abort 用於中止請求並返回錯誤狀態碼

from linebot import (
    LineBotApi, WebhookHandler # LineBotApi 和 WebhookHandler 是用於與 Line 平台互動的核心模組
)

# 引入 linebot 異常處理
from linebot.exceptions import (
    InvalidSignatureError # InvalidSignatureError 是一種異常處理，用於捕捉簽名驗證錯誤
)
# 引入 linebot 訊息元件
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, # MessageEvent, TextMessage, TextSendMessage 是一些模型，用於處理和回應特定類型的訊息
)

app = Flask(__name__) # Flask 應用程序實例

# LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 類似聊天機器人的密碼，記得不要和他人分享。
line_bot_api = LineBotApi('') 
# Channel access token 來自 Messaging API settings

handler = WebhookHandler('') # Channel secret 來自 Basic settings
# WebhookHandler 用於處理來自 Line 平台的 Webhook 請求


# 此為 Webhook callback endpoint
@app.route("/callback", methods=['POST']) # 定義了一個路由，當收到 /callback 的 POST 請求時，會調用 callback 函數
def callback():
    signature = request.headers['X-Line-Signature'] # 從請求標頭中取得 X-Line-Signature
    # 這是 Line 平台用來驗證請求是否真實的簽名

    body = request.get_data(as_text=True) # 取得請求的原始數據，並轉換為文字格式

    # 將請求的內容和簽名交給 WebhookHandler 處理。如果簽名無效，會引發 InvalidSignatureError 異常
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400) # 當捕捉到 InvalidSignatureError 時，程式會印出錯誤訊息並返回 400 錯誤狀態碼

    return 'OK' # 如果處理成功，返回 OK 字符串


# decorator 負責判斷 event 為 MessageEvent 實例，event.message 為 TextMessage 實例。
# 所以此為處理 TextMessage 的 handler
@handler.add(MessageEvent, message=TextMessage) # 當接收到 MessageEvent 且訊息類型是 TextMessage 時，這個函數會被調用
def handle_message(event):
    # 決定要回傳什麼 Component 到 Channel，這邊使用 TextSendMessage
    line_bot_api.reply_message( # line_bot_api.reply_message 用於回應使用者的訊息
        event.reply_token, # event 是一個事件對象，包含了事件的所有信息
        # event.reply_token 是回應這個訊息所需的令牌
        TextSendMessage(text=event.message.text)) # # event.message.text 為使用者的輸入，把它原封不動回傳回去

# __name__ 為內建變數，若程式不是被當作模組引入則為 __main__
if __name__ == "__main__": # 這行檢查當前模塊是否是主程序模塊。如果是，則執行後續代碼
    # 運行 Flask server
    app.run()

'''
1. 簽名 (Signature)
簽名在這裡指的是 X-Line-Signature。
當 Line 伺服器向我們的應用程序發送 Webhook 請求時，它會在請求的標頭中包含一個簽名。
這個簽名是用來驗證請求真實性的重要信息。
簽名的作用類似於數字簽名，確保請求確實是來自 Line 伺服器，而不是偽造的。

    1. 簽名驗證的步驟：
        1. 簽名生成：
            Line 伺服器在發送請求時，使用 CHANNEL_SECRET 進行 HMAC-SHA256 哈希運算生成簽名，並將其包含在請求的標頭中。
        
        2. 簽名驗證：
            我們的應用程序收到請求後，提取請求中的 X-Line-Signature。
            使用相同的 CHANNEL_SECRET 和請求體 (body)，進行相同的 HMAC-SHA256 運算生成簽名。
            將生成的簽名與請求中的簽名進行比對。如果兩者相同，則請求被認為是有效的。
            這樣的驗證過程確保了只有真正來自 Line 伺服器的請求才能被處理，避免了偽造請求的風險。

            
2. 令牌 (Token)
令牌在這裡指的是 CHANNEL_ACCESS_TOKEN。
這是一個機密的字符串，用於授權我們的應用程序調用 Line API。它相當於應用程序的身份憑證。

    1. 令牌的使用：
        1. 請求授權：            
            每當我們的應用程序需要調用 Line API (例如發送訊息)時，都需要在請求中包含這個令牌。
            Line 伺服器會驗證這個令牌，以確保請求來自授權的應用程序。

    2. 身份驗證：
        令牌的使用確保了只有經授權的應用程序才能調用 Line API，從而保護用戶資料和服務的安全性。

'''

# =================================================================================================

# 引入套件 flask
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
# 引入 linebot 異常處理
from linebot.exceptions import (
    InvalidSignatureError
)
# 引入 linebot 訊息元件
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage,
    StickerSendMessage, VideoSendMessage
)

app = Flask(__name__)

# LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 類似聊天機器人的密碼，記得不要放到 repl.it 或是和他人分享
line_bot_api = LineBotApi('')
handler = WebhookHandler('')


# 此為 Webhook callback endpoint
@app.route("/callback", methods=['POST'])
def callback():
    # 取得網路請求的標頭 X-Line-Signature 內容，確認請求是從 LINE Server 送來的
    signature = request.headers['X-Line-Signature']

    # 將請求內容取出
    body = request.get_data(as_text=True)

    # handle webhook body（轉送給負責處理的 handler，ex. handle_message）
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

# decorator 負責判斷 event 為 MessageEvent 實例，event.message 為 TextMessage 實例。所以此為處理 TextMessage 的 handler
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    reply_message = TextSendMessage(text='請輸入正確指令')

    # 根據使用者輸入 event.message.text 條件判斷要回應哪一種訊息
    if user_message == '圖片':
        reply_message = ImageSendMessage(
             original_content_url='https://i.imgur.com/dDHKEjn.jpeg',
                preview_image_url='https://i.imgur.com/dDHKEjn.jpeg'
            )
    elif user_message == '貼圖':
        # pass 為 Python 內部關鍵字，主要為佔位符號，待之後再補充區塊程式邏輯而不會產生錯誤
        reply_message = StickerSendMessage(
            package_id='1070',
            sticker_id='17840'
            )
    elif user_message == '影片':
        reply_message = VideoSendMessage(
            original_content_url='https://videos.pexels.com/video-files/11856385/11856385-sd_540_960_25fps.mp4',
            preview_image_url='https://i.imgur.com/A7RiNzD.png'
            )
    else:
        reply_message = TextSendMessage(text=event.message.text)

    line_bot_api.reply_message(
        event.reply_token,
        reply_message)
# __name__ 為內建變數，若程式不是被當作模組引入則為 __main__
if __name__ == "__main__":
    # 運行 Flask server
    app.run()

# =================================================================================================

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, StickerSendMessage,
    AudioSendMessage, VideoSendMessage, LocationSendMessage, TemplateSendMessage,
    ButtonsTemplate, PostbackAction, MessageAction, URIAction
)
'''
1. 文字回應訊息物件
    TextSendMessage 使用方式如下，其中 text 參數為文字字串
'''
text_message = TextSendMessage(text='Hello, world')


'''
2. 圖片回應訊息物件
    ImageSendMessage 為圖片回應訊息物件，
        1. original_content_url 為圖片網址(原圖)
        2. preview_image_url 為預覽圖片網址。
'''
image_message = ImageSendMessage(
    original_content_url='https://example.com/original.jpg',
    preview_image_url='https://example.com/preview.jpg'
)

'''
3. 貼圖回應訊息物件
    StickerSendMessage 為貼圖回應訊息物件，
        1. package_id(STKPKGID)為貼圖的系列 id
        2. sticker_id (STKID)為貼圖的 id。
'''
sticker_message = StickerSendMessage(
    package_id='1',
    sticker_id='1'
)

'''
4. 音檔回應訊息物件
    AudioSendMessage 為聲音回應訊息物件
        1. original_content_url 為音檔網址
        2. duration 時間長度。
'''
audio_message = AudioSendMessage(
    original_content_url='https://example.com/original.m4a',
    duration=240000
)

'''
5. 影片回應訊息物件
    VideoSendMessage 為聲音回應訊息物件
        1. original_content_url 為影音檔網址
        2. preview_image_url 為預覽圖需要提供圖片檔案網址)。
'''
video_message = VideoSendMessage(
    original_content_url='https://example.com/original.mp4',
    preview_image_url='https://example.com/preview.jpg'
)

'''
6. 地點回應訊息物件
    LocationSendMessage 為地點回應訊息物件
        1. title 為標頭
        2. address 為地址
        3. latitude 緯度
        4. longitude 經度。
'''
location_message = LocationSendMessage(
    title='my location',
    address='Tokyo',
    latitude=35.65910807942215,
    longitude=139.70372892916203
)

'''
7. 模版回應訊息物件
    TemplateSendMessage 模版訊息可以搭配 ConfirmTemplate 提供確認選項，讓使用者可以選擇並執行對應的行動。

        1. PostbackAction 可以送出資料到 Webhook 其中的 data 會是攜帶的訊息資料，讓我們可以知道使用者的意圖進行邏輯判斷
        2. 而 MessageAction 則是會幫使用者發出文字訊息，例如點選後送出：message text。
        3. URIAction 則會開啟網址的頁面
'''
buttons_template_message = TemplateSendMessage(
    alt_text='Buttons template',
    template=ButtonsTemplate(
        thumbnail_image_url='https://example.com/image.jpg',
        title='Menu',
        text='Please select',
        actions=[
            PostbackAction(
                label='postback',
                display_text='postback text',
                data='action=buy&itemid=1'
            ),
            MessageAction(
                label='message',
                text='message text'
            ),
            URIAction(
                label='uri',
                uri='http://example.com/'
            )
        ]
    )
)

# =================================================================================================

'''
當使用者選擇:

    1. 查詢股價 
        選單會自動產生 【@查詢股價】文字，並回傳【請問你要查詢的股票是？】文字
        當使用者輸入該股票代號後回傳股價資訊

    2. 報明牌   
        選單則會產生【報明牌】文字並隨機產生一個股票代碼和公司名稱回傳 (當然可以撰寫自己的報明牌邏輯)
    
    3.  【@查詢股價】選單後又選擇 【@報明牌】 
        則下次還需要再次選擇【@查詢股價】才能輸入股票代碼
'''


import random

# 引入套件 flask
from flask import Flask, request, abort


from linebot import (
    LineBotApi, WebhookHandler
)
# 引入 linebot 異常處理
from linebot.exceptions import (
    InvalidSignatureError
)
# 引入 linebot 訊息元件
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, StickerSendMessage
)

# 當作報明牌隨機的股票 list
good_luck_list = ['2330 台積電', '2317 鴻海', '2308 台達電', '2454 聯發科']
# 範例股價資訊，可以自行更換成查詢股價的爬蟲資料或是即時股價查詢套件的資料
stock_price_dict = {
    '2330': 210,
    '2317': 90,
    '2308': 150,
    '2454': 300
}

# 產生 Flask 物件伺服器實例
app = Flask(__name__)

# LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 類似聊天機器人的密碼，記得不要放到 repl.it 或是和他人分享
line_bot_api = LineBotApi('')
handler = WebhookHandler('')


# 此為 Webhook callback endpoint
@app.route("/callback", methods=['POST'])
def callback():
    # 取得網路請求的標頭 X-Line-Signature 內容，會確認請求是從 LINE Server 送來的避免資訊安全問題
    signature = request.headers['X-Line-Signature']

    # 將送來的網路請求內容取出
    body = request.get_data(as_text=True)

    # handle webhook body（轉送給負責處理的 handler，ex. handle_message）
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

# decorator 負責判斷 event 為 MessageEvent 實例，event.message 為 TextMessage 實例。所以此為處理 TextMessage 的 handler
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    reply_message = TextSendMessage(text='請輸入正確指令')
    # 根據使用者輸入 event.message.text 條件判斷要回應哪一種訊息

    if user_message == '@查詢股價':
        reply_message = TextSendMessage(text='請問你要查詢的股票是？')
    elif user_message == '@報明牌':
        # random.choice 方法會從參數 list 隨機取出一個元素
        random_stock = random.choice(good_luck_list)
        reply_message = TextSendMessage(text=f'報明牌：{random_stock}')

    # 回傳訊息給使用者
    line_bot_api.reply_message(
        event.reply_token,
        reply_message)


# __name__ 為內建變數，若程式不是被當作模組引入則為 __main__
if __name__ == "__main__":
    # 運行 Flask server，預設設定監聽 127.0.0.1 port 5000（網路 IP 位置搭配 Port 可以辨識出要把網路請求送到那邊 xxx.xxx.xxx.xxx:port，app.run 參數可以自己設定監聽 ip/port）
    app.run()

# =================================================================================================

'''
當我們需要使用者連續輸入指令才能完成指定任務時，可以使用暫存資料的方式來紀錄特定使用者他上一步輸入的指令為何。
暫存的方式有很多種，例如儲存在快取記憶體或是一般的程式變數記憶體中。
這邊我們簡便使用一個 dict 當作暫存記憶體來紀錄使用者操作的指令

(有興趣可以更深入研究其他的快取記憶體方式，例如: redis、memcached 等。)

在這邊我們宣告一個【user_command_dict】dict，並將傳送訊息的使用者 id
(從 event 物件中使用 event.source.user_id 取得使用者 ID) 當作 key，dict 的 value 值則存放使用者輸入的指令。
這樣一來就可以根據每個使用者紀錄他所輸入的指令對應。
'''

'''
我們希望的功能為：
    1. 當使用者選擇【@查詢股價】選單
        回傳 請問你要查詢的股票是？ 文字，當使用者輸入該股票代號後回傳股價資訊

    2. 當使用者選擇【@報明牌】選單
        則隨機產生一個股票代碼和公司名稱回傳（可以撰寫自己的報明牌邏輯）

    3. 當使用者選擇【@查詢股價】選單後又選擇【@報明牌】 
        則下次還需要再次選擇 【@查詢股價】 才能輸入股票代碼
'''
# 使用 dict 當作使用者指令暫存空間，由於存在記憶體中，所以當程式重啟就會消失紀錄
import random

# 引入套件 flask
from flask import Flask, request, abort


from linebot import (
    LineBotApi, WebhookHandler
)
# 引入 linebot 異常處理
from linebot.exceptions import (
    InvalidSignatureError
)
# 引入 linebot 訊息元件
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, StickerSendMessage
)

# 當作報明牌隨機的股票 list
good_luck_list = ['2330 台積電', '2317 鴻海', '2308 台達電', '2454 聯發科']
# 範例股價資訊，可以自行更換成查詢股價的爬蟲資料或是即時股價查詢套件的資料
stock_price_dict = {
    '2330': 210,
    '2317': 90,
    '2308': 150,
    '2454': 300
}

# 產生 Flask 物件伺服器實例
app = Flask(__name__)

# LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 類似聊天機器人的密碼，記得不要放到 repl.it 或是和他人分享
line_bot_api = LineBotApi('')
handler = WebhookHandler('')


# 此為 Webhook callback endpoint
@app.route("/callback", methods=['POST'])
def callback():
    # 取得網路請求的標頭 X-Line-Signature 內容，會確認請求是從 LINE Server 送來的避免資訊安全問題
    signature = request.headers['X-Line-Signature']

    # 將送來的網路請求內容取出
    body = request.get_data(as_text=True)

    # handle webhook body（轉送給負責處理的 handler，ex. handle_message）
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


# 使用 dict 當作使用者指令暫存空間，由於存在記憶體中，所以當程式重啟就會消失紀錄
user_command_dict = {}

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    reply_message = TextSendMessage(text='請輸入正確指令')
    user_id = event.source.user_id
    # 可以印出來看看 user_id
    print('user_id', user_id)

    # 根據使用者 ID 暫存指令
    user_command = user_command_dict.get(user_id)
    print('user_command', user_command)

    # 判斷使用者輸入為 @查詢股價 且 之前輸入的指令非 @查詢股價
    if user_message == '@查詢股價' and user_command != '@查詢股價':
        reply_message = TextSendMessage(text='請問你要查詢的股票是？')
        user_command_dict[user_id] = '@查詢股價'
    elif user_message == '@報明牌':
        random_stock = random.choice(good_luck_list)
        reply_message = TextSendMessage(text=f'報明牌：{random_stock}')
        user_command_dict[user_id] = None
    # 若上一個指令為 @查詢股價
    elif user_command == '@查詢股價':
        stock_price = stock_price_dict[user_message]
        if stock_price:
            reply_message = TextSendMessage(text=f'成交價：{stock_price}')
            # 清除指令暫存
            user_command_dict[user_id] = None

    # 回傳訊息給使用者
    line_bot_api.reply_message(
        event.reply_token,
        reply_message)
    '''
    若需要取得使用者資料 
        (例如: 顯示名稱、照片網址等)
        可以使用: line_bot_api.get_profile(user_id) 傳入使用者 ID 當作參數
    '''
    # 取得使用者資料
    profile = line_bot_api.get_profile(user_id)
    print(profile.display_name)
    print(profile.user_id)
    print(profile.picture_url)
    print(profile.status_message)

if __name__ == "__main__":
    app.run()





