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
本篇將運用到Logistic Regression、 Random Forest、 Naive Bayes 這三個模型，來進行道瓊工業指數的漲跌預測
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
1. Naive Bayes:
    Naive Bayes分類器是一種基於貝葉斯定理的概率分類器。
    它假設特徵之間是相互獨立的，即在給定類別的情況下，一個特徵的存在與其他特徵的存在無關。
    雖然這個假設在實際中很難成立，但Naive Bayes分類器在很多應用中仍然能夠取得不錯的效果，特別是在文本分類和垃圾郵件過濾等問題上。

    
2. Naive Bayes 調參:
    alpha= 1.8       # alpha 是一個平滑參數，通常用於處理概率估計中的零概率問題。這個平滑技術被稱為拉普拉斯平滑或加法平滑 (Additive Smoothing)

    在 Naive Bayes 分類中，我們根據訓練數據計算特徵在各個類別中的條件概率。
    如果某個特徵在某類別的訓練數據中從未出現過，則該特徵的條件概率會被計算為零。
    這會導致最終的類別概率也變為零，從而影響分類結果。

    為了解決這個問題，我們引入了平滑技術。
    拉普拉斯平滑是最常用的一種方法。alpha 是平滑參數，用於調整概率估計中的每一項，防止出現零概率。
        P(x|y) = N(x,y)+alpha / N(y)+(alpha*K)
            N(x,y): 是在類別y中特徵x出現的次數。
            N(y):  是類別y中所有特徵出現的總次數。
            K: 是特徵的總數 (對於多項式分佈來說，K是特徵的可能取值數)。
            alpha: 是平滑參數

        當 alpha > 0時 ，每個條件概率都會有一個非零的基礎值，即使該特徵在訓練數據中從未出現過。
            
            1. alpha = 1: 這是拉普拉斯平滑的默認值，通常效果良好。
            2. alpha < 1: 稱為過平滑，會減少極端概率，但在某些情況下可能導致偏差。
            3. alpha > 1 :稱為欠平滑，對概率估計的影響較大，會增加極端概率的值。
'''
from sklearn.naive_bayes import MultinomialNB # 從scikit-learn庫中導入多項式貝式分類器MultinomialNB

# Random Search
nb_random_grid = { # 參數空間
  'alpha': [x for x in np.arange(0.001,100,0.01)] # 
} 

nb_model = MultinomialNB() # 貝式分類器實體

nb_random_search = RandomizedSearchCV(
  estimator=nb_model, # 調參model
  param_distributions=nb_random_grid, # 參數空間
  n_iter = 100, # 迭代次數
  scoring='accuracy', # 準確率
  cv = 3, # 交叉驗證摺數
  verbose=2, # 詳細程度
  random_state=42, # 隨機種子
  n_jobs=-1 # 可用cpu
)

nb_random_search.fit(final_traindata_tfidf, train["Label"]) # 調參開始(訓練集)

nb_result = nb_random_search.fit(final_traindata_tfidf, train["Label"])  # 調參(測試集)

nb_random_model = nb_random_search.best_estimator_ # 調參後最佳model

print("Scroe of train set: % .10f" % (nb_random_model.score(final_traindata_tfidf, train["Label"])))
print("Scroe of test set: % .10f" % (nb_random_model.score(final_testdata_tfidf, test["Label"])))
print("Best score:{}".format(nb_random_search.best_score_))
print("Best parameters:{}".format(nb_random_search.best_params_))


# Grid Search
nb_param_grid= {'alpha': [x for x in np.arange(0.1, 80, 0.1)]}
nb_model = MultinomialNB()
nb_grid_search = GridSearchCV(nb_model, nb_param_grid, scoring='accuracy', cv=5)
nb_result = nb_grid_search.fit(final_traindata_tfidf, train["Label"])

nb_grid_model = nb_grid_search.best_estimator_

print("Scroe of train set: % .10f" % (nb_grid_model.score(final_traindata_tfidf, train["Label"])))
print("Scroe of test set: % .10f" % (nb_grid_model.score(final_testdata_tfidf, test["Label"])))
print("Best score:{}".format(nb_grid_search.best_score_))
print("Best parameters:{}".format(nb_grid_search.best_params_))


# 最終結果
nb_model = MultinomialNB(alpha=1.8)
nb_model.fit(final_traindata_tfidf, train["Label"])

train_pred = nb_model.predict(final_traindata_tfidf)
test_pred = nb_model.predict(final_testdata_tfidf)

train_accuracy = accuracy_score(train['Label'], train_pred)
test_accuracy = accuracy_score(test['Label'], test_pred)
print("Accuracy of train set: {:.4f}".format(train_accuracy))
print("Accuracy of test set: {:.4f}".format(test_accuracy))

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