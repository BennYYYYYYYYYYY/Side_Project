# =================================================================================================
# 甚麼是github
'''
1. 什麼是Git?
    Git 是一個【分散式版本控制軟體系統】，最初由 Linus Torvalds 所開發 (也是作業系統 Linux 系統的開發者)
    其最初目的是為更好地管理 Linux kernel 開發而設計。現在 Git 已經成為主流的版本控制系統。

2. 什麼是Github
    Github 則是一個支援 git 程式碼存取和遠端托管的網路平台服務，
    讓使用者可以在瀏覽器使用 UI 的方式來進行 Git 版本管理相關操作，但其實其背後就是一個 Git 版本控管伺服器。
    目前有許多的開放原始碼的專案都是使用 Github 進行程式碼的管理，(例如: Python 網路請求套件 Requests)。
    若未來有志於從事程式設計和軟體開發相關工作的話，建議可以熟悉掌握Git、Github，並建立 Github Profile 作品集。

3. 為什麼需要版本控制系統?
    想像使用 Microsoft Office 和同事合作專案的情境，當修改了一個新的版本的檔案後，
    可能需要在檔名後面多加了日期或是版本號，然後透過 email 寄出給其他同仁。
    隨著修改次數的增多，檔案會越來越多，若是檔名結構沒有命名好的話，要找到想要的版本，往往會是一場災難。
    
    尤其是在軟體開發這種需要常常更新版本和協同合作的過程，我們需要更好的工具來幫我們管理程式碼的版本，
    這時我們就需要透過版本控制系統來強化我們的工作流程。

    一般在軟體開發中分為: 
        1. 中央式系統 (例如: Subversion、CVS 等) 
            中央式版本控制系統的工作主要在一個伺服器進行，
            由中央管理存取權限「鎖上」檔案庫中的檔案，一次只能讓一個開發者進行工作。

        2. 分散式系統 (例如: Git、BitKeeper、mercurial 等)
            分散式系統讓不同開發者直接在各自電腦上的本地檔案庫工作，並容許多個開發者同時更動同一個檔案，而每個檔案庫有另外一個合併各個改變的功能。
            分散式系統讓開發者能在沒有網路的情況下也能繼續工作 (想像當你在的咖啡店沒有網路，你仍然可以在你的本機電腦進行開發，然後將撰寫好的程式提交到本地的程式倉庫，
            等到有網路時再將程式提交到遠端的城市倉庫)，也讓開發者有充分的版本控制能力，而不需經中央管理者的許可，但分散式系統仍然可以有檔案上鎖(lock)功能。

'''

# =================================================================================================
# 上傳檔案到github

'''
將程式碼上傳到 Github Repository 程式碼倉庫上

1. 安裝 Git
    安裝完成，打開終端機輸入以下指令，若成功顯示版本，代表安裝成功
        【git --version】 

        
2. 設定帳戶
    1. 讓 Git 知道這台電腦做的修改要連結到哪一個使用者 (使用在 Github 上註冊的帳號名稱和電子信箱)
        【git config --global user.name ""】
        【git config --global user.email ""】

    2. 完成設定可以，觀看 config 是否正確
        【git config --list】

        
3. 在 Github 開啟新的 Repository
    設定 public or private

    
4. 串連本地資料夾和雲端程式碼倉庫
    串連本地資料夾，把我們的資料夾下的程式碼推送上去
        1. 透過終端機 echo 指令新增一段字串 "# demo-github" 檔案到 README.md。
            【>>】為 append 的意思 (若還記得檔案處理的【a】是新增到檔案最後的意思)
            README.md 是這個專案的說明檔案，建議每個專案都可以寫上，語法是 markdown 語法 .md 檔案
            
            這行命令的作用是將字符串 "# demo-github" 作為一個新的標題，追加到名為 README.md 的文件末尾。
            如果這個文件不存在，該命令會創建一個新的 README.md 文件並將內容寫入其中。
                【echo "# demo-github" >> README.md】

                
5. 接著在我們的工作資料夾下執行初始化 git
    這樣一來會把我們的資料夾變成 git 的工作資料夾
    初始化 git，這時會在資料夾下產生 .git 的資料夾儲存 git 相關資料
        【git init】

        
6. 使用 【git status】 觀看目前狀況
    這個命令會告訴你哪些文件已經被修改、哪些文件在暫存區中，以及哪些文件尚未被追蹤或提交。

     
7. 接著將 README.md 這個檔案加入暫存區域
   當準備將文件提交到 Git 存儲庫時，需要先將文件從工作區域 (Working Directory) 添加到暫存區 (Staging Area)。
   這樣做的目的是建立一個 index (索引) 或 cache (緩存)，以便後續可以進行提交(commit)。
        
        1. git add:
            git add 是 Git 中用來將文件添加到暫存區的命令。暫存區是一個暫時的存放區域，保存你想要提交的更改。
            當你對工作區中的文件進行更改後，需要使用 git add 命令將這些更改放入暫存區，準備進行提交。
        
        2. README.md:
            README.md 是你要添加到暫存區的具體文件名稱。在這個例子中，README.md 文件是你之前創建並編輯的文件。
  
        若不想被 git 追蹤的檔案 (ex. 不想被 git 追蹤，上傳到 github 的檔案)，可以把檔案路徑寫在 .gitignore 這個檔案中。
        (放入 .gitignore 的檔案路徑就不會被列入 git 追蹤，例如機密文件就不會不小心推送上去網路雲端上)
        
    # add 將檔案從工作區域放到 staging Area，若有多個檔案可以使用【git add .】 但一般建議一個個檔案加入，避免加入不必要的檔案
    【git add README.md】

                    
8. 提交檔案到本地倉庫
     # commit 是將檔案放到本地端倉庫 -m 為這次提交的描述訊息 message 縮寫
        【git commit -m "first commit"】
     
        
9. 設定要提交本地端程式到遠端 Github 的位址設定
    【git remote add】origin 是一般常見命名，意思是遠端的程式庫為多人合作時大家使用的最原版的程式碼 (可命不同名稱)

    # git remote add 將本地端與雲端做串連，origin 為設定遠端的名稱，後面是遠端的網 (用自己 Github repo 程式庫的網址)
    【git remote add origin https://github.com/xxxxxx/xxxxx.git】

            
10. 將本地端倉庫的程式碼推送到雲端上
    # git push 是將本地端倉庫推送到雲端上，-u 是設定預設使用 origin 為遠端，只要第一次 push 時加就好。branch 為主幹 master (若是跟其他工程師合作，會切出 checkout 自己的 branch)
        【git push -u origin master】

'''

'''
成功上傳
    若需要和他人協作專案可以在專案新增夥伴 collaborator
    在右上角進入 setting 可以選擇到 collaborator 新增合作者 email 就有修改該專案的權限
'''

# =================================================================================================
# 情境問題

'''
1. 情境：不想加入的新檔案不小心加入暫存區?
    1. 尚未有任何 commit  
        【git rm --cached <file>】

    2. 已經有 commit 過
        【git reset HEAD <file>】

            1. HEAD 代表目前分支指到的 commit 節點位置的辨識符號
            2. HEAD^ 或 HEAD^1 代表上一個 commit 節點
            3. HEAD^2 代表上上個節點。

    所以當尚未 commit 之前沒有產生 HEAD 所以就使用 git rm --cached <file> 而非 git reset HEAD <file>
    

2. 情境：若不小心已經 commit 想要反悔？
    若我們不小心 commit 了不想要的修改，我們可以透過【git reset】來修正。
    先來認識一下 Reset 的模式。根據不同參數會有不同結果。
        1. mixed (預設)
        2. soft  
        3. hard 
    【git reset HEAD^1 --模式】


3. 情境：如何將開發新功能分支 branch 合併到主幹 master 分支(主要的程式版)上？
    1. 先確認我們的程式是在主幹 master 上 (on master)
        【git checkout master】

    2. 從遠端 origin 抓取最新的程式碼並合併到本地端 master
        【git pull origin master】

    3. 使用 【git checkout】 從主幹切出一個分支叫: new_feature
        # 使用 -b 代表建立新的 branch 分支，等同於 【git branch new_feature】 再使用 【git checkout new_feature】 移動到建立的新分支
            【git checkout -b new_feature】

'''

'''           
1. 進行程式開發，例如修改 README.md 內的 # demo-github 改為 # demo-git，此時使用 git status 可以看到不同的狀況：
    # 出現修改過的訊息
    【git status】

2. 將修改加入 staging area
    【git add README.md】

3. 提交到本地程式庫， -a 代表為 all 所有修改，-m 為 message 撰寫訊息方便日後查看
    【git commit -a -m "修改 README.md"】

4. 移動到 master 主幹分支
    【git checkout master】

5. 將 new_feature 分支合併到 master
    【git merge new_feature】

6. 刪除 new_feature branch (也可以先保留)
    【git branch -d new_feature】

7. 將合併後的 master 推送到遠端 remote repo
    【git push origin master】

'''

#================================================================================================
# 練習建議

'''
事實上，實務上 Git 指令比較常用到的只有 20%。
所以對於初學者來說先掌握 Git 的核心觀念，在遇到問題時再參考網路學習資源進一步學習會是比較好的入門方式。
另外，由於 Git 除了可以下指令碼操作外，市面上也有許多圖像化的工具可以進行使用。
然而，自己動手輸入和操作並思考背後的原理原則才是有效率的學習方式

可以在 Github 和本機電腦端建立練習用的 Git 專案，玩壞砍掉重練就好
    (把【.git】資料夾刪除再重新【git init】又是一條好漢)
'''