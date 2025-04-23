import pandas as pd
import re
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt

# 獲得資料
def get_data():
    patterns = {
        "Campaign": r"Campaign",  
        "FC": r"FC",           
        "Hospital": r"對照"     
    }


    classified_urls = {key:[] for key in patterns} # iterate dic will get key


    for root, _, filenames in os.walk(r"C:\Users\user\Desktop\每周成效報表專區\每周使用資料"):
        for file in filenames:
            for key, pattern in patterns.items():
                if re.search(pattern, file):
                    classified_urls[key].append(os.path.join(root, file))    
    
    return classified_urls