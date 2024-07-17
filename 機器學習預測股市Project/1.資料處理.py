'''
透過Reddit World News當日前25名熱門的新聞標題，預測道瓊工業指數的股價漲跌及幅度。
'''

# 資料前處理
'''
原始資料共有3個資料集
    1. reddit news：含有新聞發布的日期，及新聞標題，每日共記載25則新聞
    2. DJIA：包含日期，以及道瓊工業指數每日的開盤價、最高價、最低價、收盤價、成交量
    3. 合併資料集：記載日期、當日道瓊工業指數為漲(1)或跌(0)，與Reddit news上當日最熱門的25則新聞標題
'''

'''
1. 將所有字母轉為小寫
    為了將資料標準化，利於後續模型判讀字詞，使用lower()將字母轉成小寫 

2. 刪除數字、標點符號
    由於主要為分析新聞中的英文字詞對股市漲跌的影響
    數字及標點點符號在此先運用re.sub()刪除，並將英文保留。

3. 去除英文字母
    觀察資料可以發現每句新聞標題前方都有"b"，卻無意義。
    故運用string.ascii_lowercase將英文小寫字母整理出來，再將它們納入停用詞中，並配合第四步驟

4. 去除停用詞
    停用詞指的是a、the、is、in等英文中常見的詞，這些詞語通常對理解整個句子的語義影響較小。
    透過 stopwords.words('english')得出英文中常見的停用詞，再將他們及第三步驟提及的英文字母一併去除

5. 詞幹提取與詞形還原
    英文的名詞有複數型態，動詞則有過去式或進行式等，為了使單字標準化。
    使用 WordNetLemmatizer()將字詞變回原形。

6. 縮寫字詞的處理:
  針對原始資料某些字詞以縮寫形式呈現，如 United States 簡寫為 U.S. 之情形， 必需將這些縮寫還原回其原本型態才能在後續建立模型階段取得更好的表現。
  此處參照 Oxford English Dictionary 所提供之縮寫表與原始資料對比，最終整理出原始資料中所出現過的縮寫字詞轉換表並加以對照還原。
    
7. 片語、連續出現時產生特殊含義之詞彙的處理:
  針對此類情況，採用 n-gram 作為解決方式
    n-gram 是一種在進行 NLP 時經常使用到的處理方法。
    n-gram 是一種將句子以不同長度切分為各個字詞的方法，n 即代表每次的切分長度。
      例如 「我喜歡你」 在 n = 1 的情形下會切分為 「我」 ，「喜」，「歡」，「你」
      假設 n = 2 ，即會切分為「我喜」，「喜歡」，「歡你」，以此類推。
        除此之外，一般在進行資料分析時，遇到如「我喜歡你」 此種因順序不同而具不同意義之句子時往往無法判讀其真實語意，而運用 n-gram 即可透過 「我喜」，「喜歡」 等不同欄位來釐清原句語意。
        往後若遇到原始資料中有四字成語、三字聯詞或兩字片語的情形，也可以分別利用 4-gram 、3-gram、2-gram 將這些詞彙切分出來

'''
import os # 用於操作系統層級的功能和正則表達式操作
import re # # 用於操作系統層級的功能和正則表達式操作
import nltk # 自然語言處理工具包
import string # 包含一些字符串操作的常量，例如標點符號
import pandas as pd # 數據處理和分析工具，用於處理結構化數據
import numpy as np # 數值計算工具
from nltk.corpus import stopwords # NLTK 提供的停用詞表
from nltk.tokenize import word_tokenize # 用於將句子分詞成單詞
from nltk.stem.wordnet import WordNetLemmatizer # 詞形還原工具
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # 將文本轉換為數值特徵(TF-IDF)
from sklearn.linear_model import LogisticRegression # 邏輯迴歸模型
from sklearn.model_selection import train_test_split # 用於拆分數據集
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # 評估指標
from datetime import timedelta, datetime # 用於日期和時間操作

# 下載必要的 NLTK 資源
nltk.download('punkt') # 用於分詞的模型
nltk.download('stopwords') # 停用詞表
nltk.download('wordnet') # 詞形還原工具

# 讀取資料: 讀取 CSV 文件並將其轉換為 DataFrame
data = pd.read_csv(r'C:\Users\user\Desktop\Python\機器學習預測股市project\Daily news dataset\Combined_News_DJIA.csv')

def preprocess(processdata):

    # 創建了一個名為 headlines 的列表，包含 'Top1' 到 'Top20' 的字符串
    headlines = []
    for i in range(1, 21):
      headlines.append('Top'+str(i))

    # 把新聞標題轉換為小寫
    processdata[headlines] = processdata[headlines].astype(str)
    processdata[headlines] = processdata[headlines].applymap(str.lower) # applymap(str.lower) 方法對 DataFrame 的每一個值應用 str.lower() 函數
    
    # 組成以天為單位的data
    processdata_headlines = []
    for row in range(0,len(processdata.index)): # len(processdata.index) 返回 DataFrame 的行
      processdata_headlines.append(' '.join(str(x) for x in processdata.iloc[row,2:27])) # processdata.iloc[row, 2:27] 選取第 row 行的第 3 到第 27 列, str(x) for x in ... 將這些值轉換為字符串，然後用 ' '.join(...) 將這些字符串用空格連接起來

    # 刪除數字與標點符號
    for line in range(len(processdata_headlines)):
      processdata_headlines[line] = re.sub(r'[^A-Za-z]'," ", processdata_headlines[line]) 
      # 正則表達式 re.sub 將所有非字母字符替換為空格正則表達式,  [^A-Za-z] 匹配所有非字母字符（包括數字和標點符號）
      # re.sub(pattern, repl, string): 用於將 string 中所有匹配 pattern 的子字符串替換為 repl。這裡 pattern 是正則表達式，repl 是替換字符串，string 是要處理的原始字符串。
      # processdata_headlines[line]，即更新 processdata_headlines 列表中的第 line 行文本

    
    # 將標題切為個別字詞
    for sentence in range(len(processdata_headlines)):
      processdata_headlines[sentence] = word_tokenize(processdata_headlines[sentence]) # word_tokenize 函數將每個合併的新聞標題分割成單詞列表


    # 去除英文字母及停用詞
    alpha = []
    for abc in string.ascii_lowercase : # string.ascii_lowercase 是一個包含所有小寫字母的字符串
      alpha.append(abc)  # 通過 for 循環將每個小寫字母添加到 alpha 列表中
    en_stops = stopwords.words('english') # 獲取 NLTK 提供的英語停用詞列表
    en_stops.extend(alpha) # extend 是列表方法，用於將另一個列表中的所有元素添加到當前列表中, 將 alpha 列表中的所有小寫字母添加到 en_stops 列表中
    ''' 現在的 en_stop中包含【英文停用詞列表】+【字母】'''
    for sentence in range(len(processdata_headlines)):
      processdata_headlines[sentence] = [w for w in processdata_headlines[sentence] if w not in en_stops] # 使用list cemprehension過濾掉停用詞和單個字母
      # [表達式 for 項目 in 可迭代對象 if 條件]: 【表達式】是要添加到新列表中的元素, 【for 項目 in 可迭代對象】是一個循環，遍歷可迭代對象中的每個項目,  【if條件】只有在條件為真時，才會將項目添加到新列表中
    
    
    # 詞幹提取與詞形還原
    for sentence in range(len(processdata_headlines)): # 使用 WordNetLemmatizer 進行詞形還原
      processdata_headlines[sentence] = [WordNetLemmatizer().lemmatize(w) for w in processdata_headlines[sentence]] # WordNetLemmatizer().lemmatize(w): 將單詞 w 還原為其基本形式(默認為名詞形式)
      processdata_headlines[sentence] = [WordNetLemmatizer().lemmatize(w, pos='v') for w in processdata_headlines[sentence]] # 進行動詞形式的詞形還原，pos='v' 指定詞性為動詞。 
    ''' 第一行對【名詞】進行還原, 第二行對【動詞】進行還原
            1. 動詞 "running" 的詞根是 "run"。
            2. 名詞 "better" 的詞根是 "good"。
    '''


    # 將單詞列表重新轉換為便於文本分析的形式
    final_processdata_headlines = [] # 每個新聞標題最終會被重新組合成一個字符串，並添加到這個列表中
    for words in processdata_headlines :
      filter_words = "" # 創建了一個空字符串 filter_words, 這個字符串最終會包含所有單詞，中間用空格分隔
      for i in range(len(words)) :
        filter_words = filter_words + words[i] + " "
      final_processdata_headlines.append(filter_words)  
      '''
      1. 假設有一個包含三天新聞標題的數據集，經過前面步驟的預處理後，processdata_headlines 的結構如下:

            processdata_headlines = [
                ["apple", "stock", "rise", "after", "new", "product", "launch"],
                ["market", "down", "due", "to", "economic", "uncertainty"],
                ["technology", "sector", "shows", "signs", "of", "recovery"]
            ]
        
        2. words第一次迭代: 

            words = ["apple", "stock", "rise", "after", "new", "product", "launch"]
            
        3. 遍歷 words 中的每個單詞，並將其添加到 filter_words 中：

            epoch 1 : filter_words = "apple "
            epoch 2 : filter_words = "apple stock "
            epoch 3 : filter_words = "apple stock rise "

        4. 完整的句子再 append 到 final_processdata_headlines 中:

            final_processdata_headlines = [
                "apple stock rise after new product launch ",
                "market down due to economic uncertainty ",
                "technology sector shows signs of recovery "
            ]

      '''
    return final_processdata_headlines 


# 將數據集按日期劃分為訓練集和測試集，並對這兩個數據集進行預處理
train = data[data['Date'] < '2015-01-01'] # 將所有日期早於 2015 年 1 月 1 日的數據選擇為訓練集
# data 是包含所有數據的 DataFrame, data['Date'] 取出 data 中 Date 列的所有值
# 如果 Date 列的值早於 2015 年 1 月 1 日，則為 True，否則為 False
# 結果是生成一個新的 DataFrame，包含所有早於 2015 年 1 月 1 日的數據。

test = data[data['Date'] > '2014-12-31'] # 結果是生成一個新的 DataFrame，包含所有晚於 2014 年 12 月 31 日的數據。
'''
日期字符串格式:
  ISO 8601 日期格式(YYYY-MM-DD)的字符串比較結果與實際日期比較結果一致。
  這意味著字符串 '2014-12-31' 會被認為小於字符串 '2015-01-01'，這符合我們對日期的直觀理解。

pandas 的智能比較:
  pandas 能夠智能地處理日期時間數據，即使 Date 列中的數據類型是 datetime。
  pandas 也能正確地進行比較。
'''

# 預處理訓練/測試集
final_traindata = preprocess(train)
final_testdata = preprocess(test)

# 此為第一天25則標題處理完，並串聯而成的句子
# print(final_traindata[0])

'''
上述處理步驟已大致將原始資料整理乾淨，但在將字詞以個別獨立的狀態切分後。
  1. 可能會忽略處理縮寫的情況。例如: United States 簡寫為 U.S.。
  2. 也可能忽略片語或是連續出現時有特殊含意，例如: nuclear power、United States 等字詞之處理。
  
接下來，必須從原始資料中所切分之所有字詞挑選出較具有【顯著意義】的字詞，減輕一些過於氾濫或過於稀少的字詞權重。
並加強具【顯著意義】字詞的權重以試圖增益機器學習模型的預測準確性。

此處使用 Scikit-learn 套件底下之TfidfVectorizer 函數為我們進行字詞權重的分配。
TfidfVectorizer 是一個基於 TF-IDF 指標所開發出的字詞處理函數， 
而 TF-IDF 則是一種考量各字詞在原始資料中的密度、稀少性、頻率後將字詞給予權重評分的一個指標。

  在自然語言處理中，常利用TF-IDF法來【找到關鍵字】作分析使用
  TF-IDF法的全名是 Term Frequency - inverse Document Frequency。
  簡單來說，他是一種篩選關鍵字的方法，也是現在常使用的文字加權技術，主要有以下兩類：

    1. 字詞頻率 (Term Frequency): 計算每文章字詞頻率，我們常用詞袋稱之(bag of words)。
    2. 逆文件頻率(Inverse Document Frequency): 一個單字出現在文章數目的逆向次數。
      如果該字太常出現，就顯得不重要，例如:「你」、「我」、「他」、「了」、「吧」....這種不具有指標性的主詞或語氣詞。他的加權數值就會顯低上許多。
      會取log的原因在於隨著每個字詞的增加，如【10 - 9】與【1000 - 999】之差異。一個是1/10、一個卻是1/1000的差距。
      當文件數目越來越大時，即使出現次數的差異相同，其重要性差異變得越來越小。取對數後，可以平滑這種比例差異。
       
       TF - IDF = TF(x,y) * log(N/dfx)
          1. x: 該語詞在文章中出現的次數
          2. y: 文章中所有字詞數總和
          3. N: 文章總數量
          4. df(t) 是包含詞語t的文件數目
          5. tf(x,y): x/y，舉簡單例子來說: 「黃翔」這個字詞卻在 10 篇文章中出現過 100次，這10篇文章共有1000字，那「黃翔」的 TF 就是 100/1000=0.1
          6. IDF:文章總數/包含x的文章數量，例如: 這10篇只有1篇有包含「黃翔」，那IDF為: log(10/1)=1

          
TfidfVectorizer 能讓我們輕鬆將原始資料進行 TF-IDF 指標評分。
TfidfVectorizer 除卻能夠調配 TF-IDF 權重過高或過低之字詞，也支援一併進行 n-gram 處理。
因此我們只要使用 TfidfVectorizer 即可將原始資料以快速地將原始資料進行 n-gram 處理並給予各字詞 TF-IDF 的權重分數

  TfidfVectorizer(min_df, max_df, max_features, n_gram_range)
    1. min_df: 指定一個詞語在文檔中出現的最小頻率。如果一個詞語在少於 min_df 設定的文檔數中出現，這個詞語將會被忽略
    2. max_df: 指定一個詞語在文檔中出現的最大頻率。如果一個詞語在超過 max_df 設定的文檔數中出現，那麼詞語將會被忽略
    3. max_features: 設定要考慮的特徵(詞語)的最大數量。TfidfVectorizer 將會根據詞語的 TF-IDF 分數來選擇最重要的特徵
    4. n_gram_range: ngram_range 參數接受的是一個範圍，而不是單一的數字。表示 n-gram 的最小和最大長度
'''
tfidf_vector = TfidfVectorizer(min_df=0.01, max_df=0.99, max_features=160, ngram_range=(2, 2)) # 創建了一個 TfidfVectorizer 物件。TfidfVectorizer 是用來將文本數據轉換為 TF-IDF 特徵矩陣
# min_df=0.01: 詞語必須在至少 1% 的文件中出現，否則會被忽略
# max_df=0.99: 詞語如果在超過 99% 的文件中出現，會被忽略
# max_featuers=160: 僅保留 TF-IDF 分數最高的 160 個詞語特徵
# ngram_range(2, 2): 只採取二元語法(雙詞組)。


# 計算TF-IDF特徵值, 輸出會是一個稀疏矩陣，表示每個文件中的每個特徵詞的 TF-IDF 分數(final_traindata_tfdif)
final_traindata_tfidf = tfidf_vector.fit_transform(final_traindata) 
final_testdata_tfidf = tfidf_vector.transform(final_testdata)

'''
在機器學習中，特徵詞的選擇和特徵矩陣的生成是由 TfidfVectorizer 對象來管理的。
在對訓練數據進行 fit_transform 操作後，tfidf_vector 已經記錄了所有必要的信息，如特徵詞及其對應的 TF-IDF 分數。
因此，可以使用 tfidf_vector 來獲取特徵詞列表，而不需要直接從 final_traindata_tfidf 或 final_testdata_tfidf 中提取。

1. tfidf_vector:
  1. 擬合(fit): 擬合 TF-IDF 模型，確定學習詞語的 TF 和 IDF值，確定哪些詞語將作為特徵
  2. 轉換(transform): 將文本數據轉換為 TF-IDF 特徵矩陣，這個過程使用之前擬合好的模型
  3. 特徵詞提取: 這些詞語是模型擬合過程中選定的特徵

2. final_traindata_tfidf:
  1. 特徵矩陣: 稀疏矩陣，其中行(row)表示每個訓練文檔，列表示特徵詞(column)，矩陣中的值是每個詞語的 TF-IDF 分數
    
  EX: n_gram = 1 

      | Document    | and | bird   | cat    | dog    | 
      |-------------|-----|--------|--------|--------|
      | cat and dog | 0   | 0      | 0.176  | 0.176  |
      | dog and bird| 0   | 0.176  | 0      | 0.176  |
      | cat and bird| 0   | 0.176  | 0.176  | 0      |
'''

#印出字詞及其tfidf
terms = tfidf_vector.get_feature_names_out() #  獲取 TF-IDF 模型中的所有特徵詞, get_feature_names_out 方法返回特徵詞列表
sums = final_traindata_tfidf.sum(axis=0) # 每個句子在不同特徵值下的分數加總
data = []
for col, term in enumerate(terms):
    data.append( (term, sums[0,col] )) # 對於每個特徵詞，從 sums 矩陣中提取其總 TF-IDF 分數，並將特徵詞和分數作為一個元組添加到 data 列表中
    # 0 表示矩陣的第一行 (也是唯一一行，因為 sums 的形狀是 1 x n)，col 表示特徵詞的索引，由 enumerate(terms) 提供


ranking = pd.DataFrame(data, columns=['term','tfidf']) # 將 data 列表轉換為 DataFrame，並指定列名為 term 和 tfidf
# print(ranking) 

# 將稀疏矩陣轉換為密集矩陣
dense = final_traindata_tfidf.todense()  # todense() 方法返回一個密集的 Numpy 矩陣
denselist = dense.tolist() # 將 numpy 的密集矩陣轉換為 Python 列表
df2 = pd.DataFrame(denselist, columns=terms) # 轉換為 DataFrame，這樣可以方便地查看和分析每個文件的 TF-IDF 特徵
print(df2)

