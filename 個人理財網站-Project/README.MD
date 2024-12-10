**目標：** 製作個人理財網頁，可以記錄個人現金流與股票庫存損益

1. 使用 requests 向 API 抓取股價、兌美元匯率等數據 
2. 使用 Flask 與 sqlite3 製作網頁與連結資料庫，HTML 須放在 templates 才能顯示，圖片則是 static
3. 使用 Matplotlib 製作分析圖表
4. 網頁設計使用 bootstrap 的模板並用 flask template inheritence 調整
5. bootstrape Include via CDN 使用 Nav, Table borders, form
6. Flask 與 sqlite3 使用 g object 以及 teardown_appcontext 去做
7. 匯率API: 全球即時匯率API
8. 將後端處理的資料傳入，使用 Jinja2 (Flask 預設模型) 的方法 {{data}} 顯示在前端
9. 股價API: 證交所API
10. 圓餅圖：matplotlib.use("agg")不須顯示視窗，並且需設計後端有資料時才前端才顯示，並且當後端沒資料時進行刪除圖片，且注意Exception情況

