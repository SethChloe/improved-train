import numpy as np 
import pandas as pd 
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import seaborn as sns

# 显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 设置seaborn画图样式
sns.set(style='whitegrid',palette="muted",color_codes=True)
sns.despine(left=True,bottom=True)
sns.set(color_codes=True)
