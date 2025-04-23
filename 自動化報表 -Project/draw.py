import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from matplotlib import rcParams
import warnings

# 忽略所有警告訊息
warnings.filterwarnings("ignore")


# 設置全局字體支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False



# 繪圖(一)
def plot_and_save(figure_size, legend_size, save_path=r"Dashboard\dashboard_1.png"):

    for a, _, c in os.walk(r"表單資料\視覺化表單一"):
        sheet1_file = os.path.join(a, c[0])
    for a, _, c in os.walk(r"表單資料\視覺化表單二"):
        sheet2_file = os.path.join(a, c[0])

    df_1 = pd.read_excel(sheet1_file)
    df_2 = pd.read_excel(sheet2_file)

    df_2.drop(columns=["轉換率"], inplace=True)
    df = pd.merge(df_1, df_2, how="inner", on="日期")
    df["轉換率"] = df["診所點擊總數"] / df["訪問次數"]


    x = np.arange(len(df["日期"]))
    bar_width = 0.25
    gap = 0.1

    plt.figure(figsize=figure_size)
    plt.bar(x - bar_width / 2 - gap / 2, df["訪問次數"], width=bar_width, color="#4472C4", label="訪問次數")
    plt.bar(x + bar_width / 2 + gap / 2, df["不重複訪問者"], width=bar_width, color="#ED7D31", label="不重複訪問者")

    # 添加文本標籤
    for i in range(len(df)):
        plt.text(x[i] - bar_width / 2 - gap / 2, df["訪問次數"][i] + 50, f"{df['訪問次數'][i]:,}",
                 ha="center", va="bottom", fontsize=11)
        plt.text(x[i] + bar_width / 2 + gap / 2, df["不重複訪問者"][i] + 50, f"{df['不重複訪問者'][i]:,}",
                 ha="center", va="bottom", fontsize=11)

    # 圖例和標籤
    plt.legend(fontsize=legend_size, loc="best")
    plt.xlabel("日期", fontsize=14, labelpad=10)
    plt.ylabel("數值", fontsize=14, labelpad=10)
    plt.title("訪問次數與不重複訪問者", fontsize=legend_size)
    plt.xticks(x, df["日期"], fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()

    # 保存圖表
    print(f"Saving figure to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=100)
    plt.close()
    print(f"圖表已保存至: {save_path}")


# 繪圖(二)
def plot_dual_axis_chart(figure_size, legend_size, save_path=r"Dashboard\dashboard_2.png"):
    
    for a, _, c in os.walk(r"表單資料\視覺化表單一"):
        sheet1_file = os.path.join(a, c[0])
    for a, _, c in os.walk(r"表單資料\視覺化表單二"):
        sheet2_file = os.path.join(a, c[0])

    df_1 = pd.read_excel(sheet1_file)
    df_2 = pd.read_excel(sheet2_file)

    df_2.drop(columns=["轉換率"], inplace=True)
    df = pd.merge(df_1, df_2, how="inner", on="日期")
    df["轉換率"] = df["診所點擊總數"] / df["訪問次數"]

    fig, ax1 = plt.subplots(figsize=figure_size)

    max_clicks = df["診所點擊總數"].max() # new_add
    ax1.bar(df["日期"], df["診所點擊總數"], color="#A9C4EB", label="診所點擊總數", width=0.6)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_ylim(0, max_clicks * 1.2)  # 動態設定上限為最大值的 120%
    # ax1.set_ylim(0, 160)  # 設定 y 軸範圍

    # 添加數值標籤到柱狀圖
    for i, value in enumerate(df["診所點擊總數"]):
        ax1.text(df["日期"][i], value + 2, f"{value}", ha="center", fontsize=12, color="#2068b9")


    # 折線圖（轉換率）
    ax2 = ax1.twinx()  # 建立第二個 y 軸
    max_conversion_rate = df["轉換率"].max() # new_add
    ax2.plot(df["日期"], df["轉換率"], color="orange", label="轉換率", marker="o", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.set_ylim(0, max_conversion_rate * 1.2)  # 動態設定上限為最大轉換率的 120%
    ax2.set_yticks([i / 100 for i in range(0, int(max_conversion_rate * 100 * 1.2) + 2, 2)])
    ax2.set_yticklabels([f"{i}%" for i in range(0, int(max_conversion_rate * 100 * 1.2) + 2, 2)])
    # ax2.set_ylim(0, 0.12)  # 設定 y 軸範圍（百分比）
    # ax2.set_yticks([i / 100 for i in range(0, 13, 2)]) # 標上刻度，0~12，間隔2 
    # ax2.set_yticklabels([f"{i}%" for i in range(0, 13, 2)])

    # 添加數值標籤到折線圖
    for i, value in enumerate(df["轉換率"]):
        ax2.text(df["日期"][i], value + 0.001, f"{value:.2%}", ha="center", fontsize=11, color="black")

    # 標題與佈局調整
    plt.title("診所點擊總數與轉換率", fontsize=legend_size)
    fig.tight_layout()

    # 顯示圖例
    ax1.legend(loc="best", bbox_to_anchor=(0.8, 1), fontsize=legend_size)
    ax2.legend(loc="best", bbox_to_anchor=(0.8, 0.95), fontsize=legend_size)

    # 邊框線
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  
        plt.savefig(save_path, format='png', dpi=100)
    else:
        plt.show()

    plt.close()



# 丟到 Excel 
def pic_to_excel():
    for a, _, c in os.walk(r"成效摘要"):
        file = os.path.join(a,c[0])

    workbook = load_workbook(file)
    worksheet = workbook["Sheet1"]

    image_file_1 = r"Dashboard\dashboard_1.png"  
    img_1 = Image(image_file_1)

    image_file_2 = r"Dashboard\dashboard_2.png"  
    img_2 = Image(image_file_2)

    img_1.anchor = "I1"  
    worksheet.add_image(img_1)

    img_2.anchor = "I31"  
    worksheet.add_image(img_2)

    workbook.save(file)
    print(f"圖片已成功插入並保存到原文件：{file}")