import os
import re
import move
import sheet
import draw
import get


print("請確定是否有【上期TOP100】")
input("確認後按下任意鍵以繼續....")


# 把上週的報表移到舊資料庫
if os.listdir(r"成效摘要"):
    move.move_summry_to_old()
else:
    pass


# 這期
version=input(" 輸入日期 'YYYY.MM.DD-YYYY.MM.DD' ")
add_number=int(input(" 輸入三表距離: 30+ "))
 

# row data
classified_urls = get.get_data()


# 生成所有表單
sheet.sheet(
    version=version, 
    add_number=add_number,
    campaign=classified_urls["Campaign"][0], 
    fc=classified_urls["FC"][0], 
    hospital_list=classified_urls["Hospital"][0]
)


# 繪圖(一)
while True:
    if os.path.exists(r"Dashboard\dashboard_1.png"):
        os.remove(r"Dashboard\dashboard_1.png")

    x_1 = float(input("Figure Width Example ('10') Enter:   "))
    y_1 = float(input("Figure Height Example ('6') Enter:   "))
    legend_size = int(input("Legend Size Example '10' Enter:   "))
    figure_size = (x_1, y_1)

    draw.plot_and_save(figure_size, legend_size)
    os.startfile("Dashboard\dashboard_1.png")  # 僅適用於 Windows，顯示範例圖表

    stop = input("Save? (y/n): ").strip().lower()
    if stop != "n":
        break


# 繪圖(二)
while True:
    if os.path.exists(r"Dashboard\dashboard_2.png"):
        os.remove(r"Dashboard\dashboard_2.png")

    x_1 = float(input("Figure Width Example ('10') Enter:   "))
    y_1 = float(input("Figure Height Example ('6') Enter:   "))
    legend_size = int(input("Legend Size Example '10' Enter:   "))

    figure_size = (x_1, y_1)

    draw.plot_dual_axis_chart(figure_size, legend_size)
    os.startfile(r"Dashboard\dashboard_2.png")

    stop = input("Save? (y/n): ").strip().lower()
    if stop != "n":
        break

# 把圖加到 Excel
draw.pic_to_excel()




# 本期 top -> 上期
move.move_now_top100_to_last()

# 上期 top -> 舊檔案
move.move_last_top100_file_to_old()

# 視覺化表單 -> 舊檔案
move.move_sheet1_to_old()
move.move_sheet2_to_old()



