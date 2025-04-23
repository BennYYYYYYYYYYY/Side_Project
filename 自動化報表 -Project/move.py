import pandas as pd
import re
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt

# 可用：本期TOP -> 上期TOP
def move_now_top100_to_last():
    for a, _, c in os.walk(r"表單資料\本期TOP100"):
        file = os.path.join(a,c[0])
        destination = r"表單資料\上期TOP100"
        try:
            shutil.move(file, destination)
            print(f"成功! 已將本期TOP資料: {c[0]} 移至上期資料夾")
        except Exception as e:
            print(f"無法將本期TOP資料: {c[0]} 移至上期資料夾: {e}")


# 丟棄：上期TOP -> 舊資料
def move_last_top100_file_to_old():
    for a, _, c in os.walk(r"表單資料\上期TOP100"):
        file = os.path.join(a,c[0])
        destination = r"舊資料集合地\TOP100"
        try:
            shutil.move(file, destination)
            print(f"成功! 已將上期TOP資料: {c[0]} 移至舊資料集合地")
        except Exception as e:
            print(f"無法將上期TOP資料: {c[0]} 移至舊資料集合地: {e}")


# 丟棄：視覺化表單一 -> 舊資料
def move_sheet1_to_old():
    for a, _, c in os.walk(r"表單資料\視覺化表單一"):
        file = os.path.join(a,c[0])
        destination = r"舊資料集合地\表單一"
        try:
            shutil.move(file, destination)
            print(f"成功! 已將視覺化表單一: {c[0]} 移至舊資料集合地")
        except Exception as e:
            print(f"無法將視覺化表單一: {c[0]} 移至舊資料集合地: {e}")


# 丟棄：視覺化表單二 -> 舊資料
def move_sheet2_to_old():
    for a, _, c in os.walk(r"表單資料\視覺化表單二"):
        file = os.path.join(a,c[0])
        destination = r"舊資料集合地\表單二"
        try:
            shutil.move(file, destination)
            print(f"成功! 已將視覺化表單二: {c[0]} 移至舊資料集合地")
        except Exception as e:
            print(f"無法將視覺化表單二: {c[0]} 移至舊資料集合地: {e}")



# 丟棄：上期的成效摘要
def move_summry_to_old():
    for a, _, c in os.walk(r"成效摘要"):
        file = os.path.join(a,c[0])
        destination = r"舊資料集合地\成效摘要"
        try:
            shutil.move(file, destination)
            print(f"成功! 已將上期成效摘要: {c[0]} 移至舊資料集合地")
        except Exception as e:
            print(f"無法將上期成效摘要: {c[0]} 移至舊資料集合地: {e}")