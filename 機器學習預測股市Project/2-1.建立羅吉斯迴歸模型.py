import os
import re
import nltk
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from datetime import timedelta, datetime

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv(r"C:\Users\user\Desktop\Python\機器學習預測股市project\Daily news dataset\after_Combined_News_DJIA.csv")

def preprocess(processdata):
    # 轉小寫
    headlines = []
    for i in range(1, 26):
      headlines.append('Top'+str(i))
    processdata[headlines] = processdata[headlines].astype(str)
    processdata[headlines] = processdata[headlines].applymap(str.lower)
    
    # 組成以天為單位的data
    processdata_headlines = []
    for row in range(0,len(processdata.index)):
      processdata_headlines.append(' '.join(str(x) for x in processdata.iloc[row,2:27]))

    # remove punctuation characters
    for line in range(len(processdata_headlines)):
      processdata_headlines[line] = re.sub(r'[^A-Za-z]'," ", processdata_headlines[line])

    # 切字
    for sentence in range(len(processdata_headlines)):
      processdata_headlines[sentence] = word_tokenize(processdata_headlines[sentence]) 

    # 去除停用詞
    alpha = []
    for abc in string.ascii_lowercase :
      alpha.append(abc)      
    en_stops = stopwords.words('english')
    en_stops.extend(alpha)
    for sentence in range(len(processdata_headlines)):
      processdata_headlines[sentence] = [w for w in processdata_headlines[sentence] if w not in en_stops] 
    
    # 單字變回原形
    for sentence in range(len(processdata_headlines)):
      processdata_headlines[sentence] = [WordNetLemmatizer().lemmatize(w) for w in processdata_headlines[sentence]]
      processdata_headlines[sentence] = [WordNetLemmatizer().lemmatize(w, pos='v') for w in processdata_headlines[sentence]]   

    # 組回標題
    final_processdata_headlines = []
    for words in processdata_headlines :
      filter_words = ""
      for i in range(len(words)) :
        filter_words = filter_words + words[i] + " "
      final_processdata_headlines.append(filter_words)  

    return final_processdata_headlines  

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
final_traindata = preprocess(train)
final_testdata = preprocess(test)

tfidf_vector = TfidfVectorizer(min_df=0.01, max_df=0.99, max_features=160, ngram_range=(2, 2))
final_traindata_tfidf = tfidf_vector.fit_transform(final_traindata)
final_testdata_tfidf = tfidf_vector.transform(final_testdata)

word = tfidf_vector.get_feature_names()
df = pd.DataFrame(final_traindata_tfidf.T.todense().transpose(), columns=word).sum(axis=0) 
#.T 是矩陣的轉置操作
# .transpose(): 將密集矩陣轉置回來
# .sum(axis=0): 計算每一列的總和，即計算每個詞在所有文檔中的 TF-IDF 值總和。

df.head() # 顯示 DataFrame 的前五個值，主要用於檢查計算結果。

# =================================================================================================


'''
完成資料處理後，便可以開始建立機器學習模型。
'''
word = tfidf_vector.get_feature_names() # 特徵名稱
input_df = pd.DataFrame(final_traindata_tfidf.todense(), columns=word) # 轉成 DataFrame 且列為特徵名
input_df.head(10) # 顯示前10項


'''
三個機器學習模型的調參，在此會運用到 scikit-learn 的調參套件來進行調參，步驟如下:

    1. 先利用 RandomizedSearchCV 得到效能尚可的參數，藉此縮小範圍
        RandomizedSearchCV 結合了隨機搜尋超參數以及交叉驗證這兩個方法
        也就是從給定的範圍中隨機挑選超參數進行模型訓練，再利用交叉驗證的方式找出平均效果最好的超參數。

    2. 再利用 GridSearchCV 仔細尋找能使效能最好的超參數
        和 RandomizedSearchCV 不同的是，GridSearchCV 會遍歷給定的範圍
        利用排列組合的方式，嘗試每一種組合，再利用交叉驗證的方式找出平均效果最好的超參數。

    # 交叉驗證

    評估模型的穩定性和泛化能力。
    例如，k折交叉驗證將數據分為k個子集，交替使用一個子集作為測試集，其餘子集作為訓練集。
'''

'''
1. Logistic Regression 模型
    Logistic Regression是一種用於分類問題的統計模型。與線性迴歸不同，它預測的是一個二元分類變數，而不是連續變數。
        1. Logistic Regression的目的是估計事件發生的概率，通常在0到1之間。
        2. Logistic Regression利用邏輯函數(Sigmoid函數)來將線性回歸的輸出轉換為概率值。

2. Logistic Regression 調參:
    penalty= 12     # 正則化方法，有"11"、"12"可選，用來防止overfitting
    C= 1            # 為正則化係數λ的倒數，數值越小代表正則化效果越強
    solver= 'lbfg'  # 最佳化演算法，共有五種可選("newton-cg", "lbfgs", "liblinear", "sag", "saga") 
    max_iter= 100   # 算法收斂最大迭代次數 

# 解釋

    1. 正則化: 

        正則化是一種防止機器學習模型 overfitting 的方法。
        正則化通過向模型的損失函數中添加一項懲罰項，限制模型參數的大小或複雜度。
        使得模型更具有泛化能力，從而提高在未見數據上的性能。

        在訓練模型時，如果模型過於複雜(如參數過多)，它可能會學到訓練數據中的噪聲，導致過擬合。
        過擬合的模型在測試數據上的表現通常不佳，因為它無法很好地泛化到新數據。正則化旨在解決這個問題。

            1.  L1 正則化 (Lasso 回歸)
                    通過向損失函數中添加權重絕對值的和來進行正則化。
                    L1正則化的特點是可以產生稀疏模型，即某些權重會被強制為零，這意味著它自動進行特徵選擇
                    【 λ∑|wi| 】
                        λ是正則化參數，用來控制正則化項的權重
                        wi是模型的參數

            2. L2 正則化 (Ridge 回歸)
                    過向損失函數中添加權重平方和來進行正則化。 
                    L2正則化會將所有權重逼近於零，但不會強制它們為零，因此它不會進行特徵選擇。 
                    【λ∑=w^2】
            
            3. Elastic Net
                Elastic Net是一種結合了L1和L2正則化的方法。
                【λ1∑wi+λ2∑wi^2】

                
    2. 最佳化演算法:

        1. Newton-CG
            適合於中等規模問題，利用二階導數信息，提高收斂速度。

        2. L-BFGS
            適合於大規模數據，利用有限記憶擬牛頓法，有效處理高維度數據。

        3. Liblinear
            針對大規模線性分類問題，效率高，適用於稀疏數據。

        4. SAG
            結合隨機和批量梯度下降的優點，適合於大規模線性模型。

        5. SAGA
            SAG的改進版本，收斂更快，適用於廣泛的大規模優化問題。
'''
from sklearn.linear_model import LogisticRegression

# 使用 RandomizedSearchCV 來尋找最佳的 Logistic Regression 模型超參數

# 設定超參數空間 (超參數空間指的是模型訓練中需要調整的超參數及其可能取值範圍)
lr_random_grid ={   # 存儲希望在隨機搜索中調整的超參數。
    'solver' : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], 
    'penalty' : ["l1", "l2"], 
    "C" : [x for x in np.arange(0.0001, 1000, 10)] ,  # 生成從 0.0001 到 1000，每隔 10 一個步長的數字列表
    "max_iter" : [int(x) for x in range(1,500,10)], # 從 1 到 500，每隔 10 一個步長的整數數字列表
    "class_weight" : ['balanced'] # 自動調整類別權重以處理不平衡數據集
}

# 創建一個 LogisticRegression 的實例，用於訓練和超參數調整
lr_random_model = LogisticRegression() 

#  設置隨機搜索 Model
lr_random_search = RandomizedSearchCV(  # RandomizedSearchCV 是 Scikit-Learn 用來進行隨機搜索交叉驗證的工具
    estimator=lr_random_model,  # 指定用於調參的基模型 (Logistic Regression)
    param_distributions=lr_random_grid, # 指定超參數空間，這裡是之前設置的字典
    n_iter = 100,  # 設定隨機搜索的迭代次數 (即從超參數空間中隨機選擇100組參數進行評估)
    scoring='accuracy',  # 指定模型評估標準 (這裡用 accuracy)
    cv = 3,  # 設置交叉驗證的折數 (這裡使用 3 折交叉驗證)
    verbose=2,  # 參數控制了在執行過程中輸出的詳盡程度，即控制信息輸出的多寡，2 輸出詳細信息，顯示每個交叉驗證折疊的進展
    random_state=42, # 設置隨機種子，用於確保結果的可重現性，這裡設置為 42
    n_jobs=-1 # 設置使用的CPU核心數，-1 表示使用所有可用的核心
)
'''
隨機種子: 
隨機數通常是由隨機數生成器生成的。這些生成的隨機數其實是偽隨機數，因為它們是由某個初始值(種子)經過一系列數學計算得出的。
當我們設定相同的初始值(種子)，就能保證每次運行程式時，生成的隨機數序列是相同的。
'''

# 使用隨機搜索(Random Search)找到的最佳超參數來訓練邏輯回歸模型，並對測試數據進行預測和評估
lr_random_search.fit(final_traindata_tfidf, train["Label"]) # 使用 final_traindata_tfidf(X) 和 train["Label"](Y) 進行隨機搜索交叉驗證，以找到最佳的超參數組合
random_lr_model = lr_random_search.best_estimator_ # best_estimator_ 屬性返回隨機搜索中找到的性能最佳的模型
predictions = random_lr_model.predict(final_testdata_tfidf) # 使用最佳模型對測試數據進行預測

# score 方法用於計算模型在給定數據和標籤上的準確率
print("Score of train set: % .10f" % (random_lr_model.score(final_traindata_tfidf, train["Label"]))) # 格式化輸出精確到小數點後10位
print("Score of test set: % .10f" % (random_lr_model.score(final_testdata_tfidf, test["Label"]))) 
print("Best score:{}".format(lr_random_search.best_score_))  
print("Best parameters:{}".format(lr_random_search.best_params_)) 



# 使用網格搜索(GridSearchCV) 來尋找最佳 Logistic Regression 模型超參數 
lr_grid ={  
    'solver' : ["newton-cg", "liblinear", "sag"],  # 優化算法選項 
    'penalty' : ["l2"],  # 正則化的方式 (L2正則化) 
    "C" : [x for x in np.arange(100, 200, 10)] ,  # 正則化強度的倒數: 100 到 200 腳步 10 的列表 
    "max_iter" : [int(x) for x in range(200, 400, 20)], # 最大迭代次數:  200 到 400 腳步 20 的整數列表 
    "class_weight" : ['balanced'] # 類別權重 (balanced: 自動處理不平衡數據集) 
} 

lr_model = LogisticRegression() 

# 設置網格搜索 
lr_grid_search = GridSearchCV(lr_model, lr_grid, scoring='accuracy') # GridSearchCV(用於調參Model, 超參數空間, 評估指標) 
lr_grid_search.fit(final_traindata_tfidf, train["Label"]) # 進行網格搜索，找出最佳的超參數組合並訓練最優模型 
  
grid_lr_model = lr_grid_search.best_estimator_  #  返回最佳性能的model 
print("Score of train set: % .10f" % (grid_lr_model.score(final_traindata_tfidf, train["Label"])))  # 套用模型計算準確率，取小數點第十位
print("Score of test set: % .10f" % (grid_lr_model.score(final_testdata_tfidf, test["Label"]))) 
print("Best score:{}".format(lr_grid_search.best_score_))    
print("Best parameters:{}".format(lr_grid_search.best_params_)) 

'''
# 使用最佳超參數訓練最終模型並評估性能
best_params = lr_random_search.best_params_ # 或 lr_grid_search.best_params_ (看選擇哪個最佳參數組合)

# 創建並使用最佳參數初始化最終模型
final_lr_model = LogisticRegression(
    solver=best_params['solver'],
    penalty=best_params['penalty'],
    C=best_params['C'],
    max_iter=best_params['max_iter'],
    class_weight=best_params['class_weight']
)
'''
# 最終結果
lr_model = LogisticRegression() 
lr_model.fit(final_traindata_tfidf, train["Label"]) # 訓練模型

train_pred = lr_model.predict(final_traindata_tfidf) # 訓練集進行預測
test_pred = lr_model.predict(final_testdata_tfidf) # 測試集進行預測

train_accuracy = accuracy_score(train['Label'], train_pred) # test['Label'] 是測試數據的真實標籤、test_pred 是測試數據的預測標籤
test_accuracy = accuracy_score(test['Label'], test_pred) # 看準確率
print("Accuracy of train set :{:.4f}".format(train_accuracy)) # 取小數點第四位
print("Accuracy of test set:{:.4f}".format(test_accuracy))


'''
1. 訓練集結果(準確率):
   Logistic Regression: 95.9% -> 63.94% 
   Random Forest: 100% -> 75.11%
   Naive Bayes: 56.36% -> 60.77%

2. 測試集結果(準確率):
   Logistic Regression: 47.88% -> 54.23% 
   Random Forest: 52.12% -> 54.23%
   Naive Bayes: 50.79% -> 53.17%

為什麼需要兩種準確率？

    1. 檢查過擬合:
        如果模型在訓練集上的準確率非常高，但在測試集上的準確率較低，這通常表示模型過擬合。
        即模型過於貼合訓練數據中的細節和噪聲，導致泛化能力差。

    2. 檢查欠擬合：
        如果模型在訓練集和測試集上的準確率都很低，這通常表示模型欠擬合，即模型沒有足夠地學習數據中的模式。

    3. 模型性能評估：  
        通過比較訓練集和測試集上的準確率，可以全面評估模型的性能和穩定性。
        理想情況下，訓練集和測試集上的準確率應該接近，且都比較高。
'''