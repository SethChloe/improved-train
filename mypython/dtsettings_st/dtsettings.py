import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats

# 显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 设置seaborn画图样式
sns.set(style='whitegrid',palette="muted",color_codes=True)
sns.despine(left=True,bottom=True)
sns.set(color_codes=True)

#一阶偏相关分析
def partial_corr(x, y, partial=[]):
    xy, xyp = stats.pearsonr(x, y)
    xp, xpp = stats.pearsonr(x, partial)
    yp, ypp = stats.pearsonr(y, partial)
    n = len(x)
    df = n - 3
    r = (xy - xp * yp) / (np.sqrt(1 - xp * xp) * np.sqrt(1 - yp * yp))
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t = (r * np.sqrt(df)) / np.sqrt(1 - r * r)
        prob = (1 - stats.t.cdf(abs(t), df)) * 2
    return r, prob

#关联分析 Apriori算法
sign = '-->'

class Apriori(object):
    def __init__(self, minsupport=0.1, minconfidence=0.4):
        self.minsupport = minsupport
        self.minconfidence = minconfidence

    def link(self, x, sign):
        x = list(map(lambda i: sorted(i.split(sign)), x))
        l = len(x[0])
        r = []
        for i in range(len(x)):
            for j in range(i, len(x)):
                if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                    r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
        return r

    def apriori(self, data):
        final = DataFrame(index=['support', 'confidence'])
        support_series = 1.0 * data.sum() / len(data)
        column = list(support_series[support_series > self.minsupport].index)
        k = 0
        while len(column) > 1:
            k = k + 1
            column = self.link(column, sign)
            sf = lambda i: data[i].prod(axis=1, numeric_only=True)
            data_2 = DataFrame(list(map(sf, column)),
                               index=[sign.join(i) for i in column]).T
            support_series_2 = 1.0 * data_2[[sign.join(i) for i in column
                                             ]].sum() / len(data)
            column = list(
                support_series_2[support_series_2 > self.minsupport].index)
            support_series = support_series.append(support_series_2)
            column2 = []
            for i in column:
                i = i.split(sign)
                for j in range(len(i)):
                    column2.append(i[:j] + i[j + 1:] + i[j:j + 1])
                cofidence_series = Series(
                    index=[sign.join(i) for i in column2])
                for i in column2:
                    cofidence_series[sign.join(i)] = support_series[sign.join(
                        sorted(i))] / support_series[sign.join(i[:len(i) - 1])]
                for i in cofidence_series[
                        cofidence_series > self.minconfidence].index:
                    final[i] = 0.0
                    final[i]['confidence'] = cofidence_series[i]
                    final[i]['support'] = support_series[sign.join(
                        sorted(i.split(sign)))]
        final = final.T.sort_values(['confidence', 'support'], ascending=False)
        return final
