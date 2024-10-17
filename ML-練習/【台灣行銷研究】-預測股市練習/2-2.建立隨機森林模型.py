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
2. Random Forest 模型
    Random Forest是一種集成學習方法，用於分類、迴歸以及其他任務。
    它通過構建多個決策樹並將它們的預測結果進行集成來提高模型的準確性和穩定性。
    這種方法克服了單個決策樹容易過擬合的缺點，並且能夠處理高維度數據和缺失值。

    Random Forest是由多個決策樹組成的集成模型。
    其核心思想是通過引入隨機性來生成多個不同的決策樹，並通過集成這些樹的結果來提高預測的準確性和穩定性。
    每棵決策樹是通過以下兩個隨機過程生成的：
        1. Bootstrap取樣:
            從原始訓練數據集中有放回地隨機抽取樣本，生成多個訓練子集。這保證了每棵樹的訓練數據不同。

        2.  隨機選擇特徵:
            在每個節點分裂時，隨機選擇特徵子集而不是使用全部特徵。這增加了樹之間的多樣性，進一步減少過擬合。
    
    # 訓練過程

        1. 構建多個決策樹:
            從訓練數據中多次進行Bootstrap取樣，每次生成一個訓練子集。
            對每個訓練子集構建一棵決策樹。每棵樹在構建過程中，僅隨機選擇部分特徵進行分裂。

        2. 集成多個決策樹:
            對於分類問題，Random Forest使用多數投票法來決定最終的分類結果。
            對於迴歸問題，則取多棵樹預測值的平均值作為最終結果。

            1. 集成學習 (Ensemble Learning)
            集成學習通過結合多個基礎模型 (稱為基學習器 Base Learners)，以期望獲得比單一模型更好的預測性能。
            集成學習的核心思想是【群體智慧】，即多個模型的協同合作能夠抵消單一模型的誤差，提高預測的準確性和穩定性。
            集成學習主要有三種方法: Bagging、Boosting、Stacking

                1. Bagging (Bootstrap Aggregating)
                    Bagging是一種通過減少模型方差來提高預測性能的技術
                    通過在原始訓練數據上隨機抽樣生成多個子數據集，對每個子數據集訓練模型，最後將這些模型的預測結果進行平均或投票。
                        方差描述模型預測的變動性。高方差模型對訓練數據非常敏感，容易過擬合，因為它捕捉了數據中的噪音。
                        方差反映模型對訓練數據細微變化的敏感程度。

                        1. Bootstrap取樣:  
                            從原始訓練數據集中有放回地隨機抽取多個子集

                        2. 訓練基學習器:
                            對每個子集訓練一個基學習器 (通常是決策樹)

                        3. 集成結果:
                            對於分類問題，使用多數投票法來決定最終分類結果；對於迴歸問題，取多個基學習器預測值的平均值。

                2. Boosting
                    Boosting是一種通過減少模型偏差來提高預測性能的技術。
                    透過逐步改進弱分類器來構建一個強分類器，每個模型在訓練時會關注前一個模型錯誤分類的樣本。
                        偏差描述模型預測的系統誤差。高偏差模型過於簡單，無法捕捉數據的複雜模式，容易欠擬合。
                        偏差反映模型預測值與真實值之間的偏離程度。

                        1. 順序訓練基學習器: 
                            依次訓練多個基學習器，每個基學習器在前一個基學習器的基礎上進行改進。
                            訓練過程中，對於被前一個基學習器錯誤分類的樣本給予更高的權重，使得後續基學習器更注重這些錯誤樣本。
                        
                        2. 集成結果:
                            將所有基學習器的結果進行加權平均或加權投票，獲得最終預測結果。

                3. Stacking
                    Stacking是一種通過組合多個不同類型的基學習器來提高預測性能的技術。
                        1. 訓練基學習器:
                            使用原始訓練數據訓練多個不同類型的基學習器 (例如決策樹、線性模型、SVM等)

                        2. 生成元特徵:
                            使用基學習器的預測結果作為新的特徵，生成一個元特徵數據集。

                        3. 訓練元學習器:
                            使用元特徵數據集訓練一個元學習器，該元學習器用來最終的預測。

3. Random Forest 調參 

    n_estimators= 138   # 森林裡樹的總數量，通常越大越好，但過大會有計算成本與overfitting風險

    max_features= auto  # 生成決策樹時，最大的特徵使用數量。【"auto"(all)、"sqrt"(√N)、"log2"(logN)】N為樣本總特徵數

    max_depth= 10       # 樹的最大深度。常在模型樣本數及特徵數較多時使用。可以防止長太多overfitting
    
    min_sample_split=7  # 控制每個內部節點至少需要包含多少樣本才能進行劃分。較大的值可以防止決策樹過度生長，從而減少過擬合的風險。
   
    min_samples_leaf=1  # 設定葉子節點中至少需要包含的最少樣本數。當葉子節點樣本數非常少時，模型容易過擬合訓練數據中的噪聲。
    
    bootstrap=True      # 建立樹時取樣的方式，若為True，表示從訓練數據中有放回地取樣(bootstrap sampling)，這樣每棵樹可能會看到不同的訓練數據。若為False，代表用全部樣本數來建立樹
    
    criterion=gini      # 分割特徵的測量方式，有"gini", "entropy" 可選
        1. "gini":使用基尼不純度(Gini impurity)作為劃分標準。基尼不純度衡量的是樣本被隨機選中的不確定性，較低的基尼不純度表示較高的純度。
        2. "entropy"：使用信息增益(Information Gain)作為劃分標準。信息增益基於信息熵，衡量的是樣本不確定性的減少，較低的熵表示較高的純度。
'''

from sklearn.ensemble import RandomForestClassifier

# Random Search
n_estimators = [int(x) for x in np.linspace(60, 160, num = 20)] # np.linspace 函數返回一個在指定範圍內等間距的數值數組
max_features = ['auto', 'sqrt']  # auto: 使用全部特徵, sqrt: 使用特徵平方根
# 設置 max_features 的用意是通過在每次分裂節點時隨機選擇部分特徵來增加模型的多樣性，從而提高隨機森林的泛化能力並減少過擬合風險
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None) # 樹的深度無限制
min_samples_split = [2, 4, 5, 7, 8, 10] # 每次分割所需的最小樣本數
min_samples_leaf = [1, 2, 3, 4, 5, 6] # 每個葉節點所需的最小樣本數
bootstrap = [True, False]
criterion = ['entropy'] # 表示在分割節點時使用 entropy 作為評估標準
random_state = [0] # 設置隨機種子

rfc_random_grid = {'n_estimators': n_estimators, # 隨機森林中決策樹數量的候選值
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'random_state':random_state,
               'criterion':criterion}

rfc = RandomForestClassifier() # 創建了一個隨機森林分類器
rfc_random_search = RandomizedSearchCV(
    estimator=rfc, # 設置要優化的模型 (rfc分類器)
    param_distributions=rfc_random_grid, # 參數搜索空間
    n_iter = 100, # 迭代100次
    scoring='accuracy', 
    cv = 3, # 交叉驗證的摺數
    verbose=2, # 詳細程度
    random_state=42, # 設置隨機種子
    n_jobs=-1 # 所有可用的處理器進行計算(-1)
)

rfc_random_search.fit(final_traindata_tfidf, train["Label"]) # 訓練數據 final_traindata_tfidf 和其標籤 train["Label"] 進行隨機搜索

rfc_random_model = rfc_random_search.best_estimator_ # 提取最佳模型

print("Score of train set: % .10f" % (rfc_random_model.score(final_traindata_tfidf, train["Label"])))
print("Score of test set: % .10f" % (rfc_random_model.score(final_testdata_tfidf, test["Label"])))
print("Best score:{}".format(rfc_random_search.best_score_))
print("Best parameters:{}".format(rfc_random_search.best_params_))



# Grid Search
n_estimators = [130,138] # 決策樹數量的候選值
max_features = ['auto'] # 每次分割時考慮所有特徵
max_depth = [5,8,10] # 最大深度候選值
max_depth.append(None) # 無最大深度限制
min_samples_split = [2,3] # 每次分割所需的最小樣本數
min_samples_leaf = [None,2,6] # 每個葉節點所需的最小樣本數
bootstrap = [True] # 使用抽樣放回
criterion = ['entropy'] # 使用entropy作為分割標準
random_state = [0] # 隨機種子候選人:0

rfc_param_grid = {    # 放入參數空間
    "random_state":random_state, 
    "max_features":max_features,
    "n_estimators":n_estimators,
    "max_depth":max_depth,
    "min_samples_leaf":min_samples_leaf,
    "min_samples_split":min_samples_split,
    "criterion":criterion
}

rfc = RandomForestClassifier() # rfc實體
rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, scoring='accuracy') 
rfc_grid_search.fit(final_traindata_tfidf, train["Label"])

rfc_grid_model = rfc_grid_search.best_estimator_ # 最佳model
print("Score of train set: % .10f" % (rfc_grid_model.score(final_traindata_tfidf, train["Label"])))
print("Score of test set: % .10f" % (rfc_grid_model.score(final_testdata_tfidf, test["Label"])))
print("Best score:{}".format(rfc_grid_search.best_score_))
print("Best parameters:{}".format(rfc_grid_search.best_params_))


# 最終結果
rfc = RandomForestClassifier(n_estimators = 138 ,criterion = 'gini' ,min_samples_split = 7, max_depth = 10, random_state=0)
rfc.fit(final_traindata_tfidf, train["Label"])

train_pred = rfc.predict(final_traindata_tfidf)
test_pred = rfc.predict(final_testdata_tfidf)

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