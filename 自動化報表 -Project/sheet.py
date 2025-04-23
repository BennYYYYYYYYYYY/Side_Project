import pandas as pd
import re
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# 忽略所有警告訊息
warnings.filterwarnings("ignore")


# 設置全局字體支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False



# 獲得列表: 本期TOP100點擊數 > 上期TOP100點擊數 = True 
def get_top100_list():
    for a, _, c in os.walk(r"表單資料\上期TOP100"):
        top100_last_file = os.path.join(a,c[0])

    for a, _, c in os.walk(r"表單資料\本期TOP100"):
        top100_now_file = os.path.join(a,c[0])

    df_1 = pd.read_excel(io=top100_last_file, usecols=["診所名稱", "點擊數"])
    df_2 = pd.read_excel(io=top100_now_file, usecols=["診所名稱", "點擊數"])

    df_1.rename(columns={"點擊數":"上期點擊數"}, inplace=True)
    df_2.rename(columns={"點擊數":"本期點擊數"}, inplace=True)


    df_merge = df_2.merge(df_1, on="診所名稱", how="left")
    df_merge.fillna(value=0, inplace=True)
    df_merge["上期點擊數"] = df_merge["上期點擊數"].astype(int)

    df_merge['bg_color'] = df_merge['本期點擊數'] > df_merge['上期點擊數']

    bg_color_list = df_merge['bg_color'].tolist()

    return bg_color_list


# 更新比較TOP的邏輯，改才比較TOP排名，而不是點擊數
def get_top100_list_update():
    for a, _, c in os.walk(r"表單資料\上期TOP100"):
        top100_last_file = os.path.join(a,c[0])

    for a, _, c in os.walk(r"表單資料\本期TOP100"):
        top100_now_file = os.path.join(a,c[0])

    df_1 = pd.read_excel(io=top100_last_file, usecols=["診所名稱", "排名"])
    df_2 = pd.read_excel(io=top100_now_file, usecols=["診所名稱", "排名"])
    df_1["排名"] = df_1["排名"].replace("TOP", "", regex=True).astype(float)
    df_2["排名"] = df_2["排名"].replace("TOP", "", regex=True).astype(int)

    df_merge = df_2.merge(df_1, on="診所名稱", how="left", suffixes=('_now', '_old'))
    df_merge.fillna(value=0, inplace=True)

    df_merge['bg_color'] = df_merge['排名_now'] < df_merge['排名_old']
    bg_color_list = df_merge['bg_color'].tolist()

    return bg_color_list




def test_to_excel(version, df_1, df_2, df_3, add_number, bg_color_list):
    full_name = f"成效摘要\LnData-HPV成效摘要{version}.xlsx"
    with pd.ExcelWriter(full_name, engine="xlsxwriter") as writer:

        workbook = writer.book # 獲取整個 excel 物件
        worksheet = workbook.add_worksheet("Sheet1")  # 創造 sheet 物件

        # 標題格式
        header_format = workbook.add_format({ # workbook.add_format(): 定義樣式
            "font_color": "white",
            "bg_color": "black",
            "font_name": "Arial",
            "align": "center",
            "font_size": 12
        })

        # 一般資料格格式
        data_format = workbook.add_format({
            "font_name": "Arial",
            "font_size": 12,
            "align": "center",
        })


        # 本期診所點擊 > 上期 格式
        data_top100_format = workbook.add_format({
            "bg_color": "yellow",
            "font_name": "Arial",
            "align": "center",
            "font_size": 12
        })


        df_list = [df_1, df_2, df_3]
        rows_list = []
        cols_list = []
        for x in df_list:
            rows, cols = x.shape
            rows_list.append(rows)
            cols_list.append(cols)
        

        add_row = 0
        df_count = 0
        for num_rows, num_cols, df in zip(rows_list, cols_list, df_list):
            for i in range(num_cols):
                    worksheet.write(add_row, i, df.columns[i], header_format)

            if df_count == 2:
                for i in range(add_row+1, add_row + num_rows+1):  
                    for j in range(num_cols):  
                        # 根據 bg_color_list [boolean] 去看是否需要應用 top100 格式
                        if bg_color_list[i - 1 - add_row]:
                            # 只應用在診所欄位
                            if j == 1: 
                                worksheet.write(i, j, df.iloc[i - 1 - add_row, j], data_top100_format)
                            else:
                                worksheet.write(i, j, df.iloc[i - 1 - add_row, j], data_format)
                        else:
                            worksheet.write(i, j, df.iloc[i - 1 - add_row, j], data_format)

            else:
                for i in range(add_row+1, add_row + num_rows+1):  
                    for j in range(num_cols):  
                        worksheet.write(i, j, df.iloc[i - 1 - add_row, j], data_format) 
            
            df_count += 1
        
            worksheet.set_column(0, num_cols - 1, 30)
            
            add_row += add_number




def sheet(version, add_number, campaign, fc, hospital_list):
    # Sheet 1
    df = pd.read_excel(campaign)
    df.drop(columns=["登陆页URL", "跳出率", "浏览量"], inplace=True)
    df.rename(columns={"访问次数":"訪問次數", "唯一身份访问者":"不重複訪問者", "平均页面停留时间":"平均停留時間（秒）"}, inplace=True)
    
    df = df.groupby("日期").agg({
    "訪問次數":["sum"],
    "不重複訪問者":["sum"],
    "平均停留時間（秒）":["mean"]    
    })

    df.reset_index(inplace=True)
    df.columns = ['日期', '訪問次數', '不重複訪問者', '平均停留時間（秒）']
    df['平均停留時間（秒）'] = df['平均停留時間（秒）'].apply(lambda x:round(x))

    try:
        df.to_excel(f"表單資料\視覺化表單一\表單一{version}.xlsx", index=False)
        print(f"成功! 表單一{version}.xlsx 已儲存")
    except Exception as e:
        print(f"表單一{version}.xlsx : {e}")


    # Sheet 2
    df_2 = pd.read_excel(fc)
    df_2 = df_2.drop(columns=["事件标签", "唯一身份访问者", "唯一事件数"])
    df_2.rename(columns={"事件总数":"診所點擊總數"}, inplace=True)

    df_2 = df_2.groupby("日期").agg({
        "診所點擊總數":["sum"]
    })

    df_2.reset_index(inplace=True)
    df_2.columns = ["日期", "診所點擊總數"]
    df_2["轉換率"] = df_2["診所點擊總數"] / df["訪問次數"]
    df_2["轉換率"] = df_2["轉換率"].apply(lambda x:f"{x:.2%}")
    try:
        df_2.to_excel(f"表單資料\視覺化表單二\表單二{version}.xlsx", index=False)
        print(f"成功! 表單二{version}.xlsx 已儲存")
    except Exception as e:
        print(f"表單二{version}.xlsx 有問題: {e}")


    # Sheet 3
    df_3 = pd.read_excel(fc)
    
    df_3.rename(columns={
    "事件总数":"點擊數",
    "事件标签":"診所"
    }, inplace=True)

    df_3.drop(columns=["日期", "唯一身份访问者", "唯一事件数"], inplace=True)
    def clean_event_label(label):
        # . 匹配任意單個字元  * 0 或多次重複之前的模式  ? 啟用非貪婪模式，意味著會匹配最少的字元，而不是盡可能多的字
        # 匹配 _ 前面的內容，如果沒有 _，則保留整個字串
        # ( ... ): 捕獲組，(?: ... ): 非捕獲組
        # ^: 從字串的開頭開始，$: 確保匹配延伸到字串的最後一個字元。
        pattern = r'^(.*?)(?:_.*)?$'
        match = re.match(pattern, label)  # match 是 re.Match Object 
        return match.group(1) if match else label # group(0) 返回配對到的原始樣子、group(1) 配對到的部分

    df_3['診所名稱'] = df_3['診所'].apply(clean_event_label)

    # 此時 groupby 的欄位會變成 index
    df_3 = df_3.groupby("診所名稱").agg({
        "點擊數":["sum"]
    })

    df_3.reset_index(inplace=True)
    df_3.columns = ["診所名稱", "點擊數"]

    df_3.sort_values(by="點擊數", ascending=False, inplace=True)
    df_3.reset_index(inplace=True, drop=True)
    df_3["排名"] = ["TOP" + str(i + 1) for i, _ in enumerate(df_3.index)]
    new_order = ["排名", "診所名稱", "點擊數"]
    df_3 = df_3[new_order]
    df_3 = df_3.head(100)

    df_join = pd.read_excel(hospital_list)
    df_use = df_3.merge(df_join, on="診所名稱", how='left').fillna(0)

    # 確認是否有縣市資料缺失    
    if any(df_use["縣市"] == 0):
        pro_list = df_use["排名"].loc[df_use["縣市"]==0].values.tolist()
        for i in pro_list:
            print(f'注意!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  排名: {i} 的縣市資料缺失!')
    else:
        pass

    try:
        df_use.to_excel(f"表單資料\本期TOP100\TOP100{version}.xlsx", index=False)
        print(f"成功! 本期 TOP100{version}.xlsx 已儲存")
    except Exception as e:
        print(f"本期 TOP100{version}.xlsx : {e}")

    top100_list = get_top100_list_update()


    try:
        test_to_excel(version, df, df_2, df_use, add_number, top100_list)
        print(f"成功! 整體摘要表格版本: {version} 成功建立")
    except Exception as e:
        print(f"摘要表格版本: {version} 有問題: {e}")