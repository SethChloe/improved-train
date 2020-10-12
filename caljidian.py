from pandas import Series,DataFrame
import pandas as pd 
import numpy as np
ret=True
print('请将该程序和需计算绩点的文件放在同一文件夹下，或直接输入文件的绝对路径。')
while ret:
    try:
        name=input('请输入需计算绩点的文件名（记得带文件名后缀！！！）：')
        df=pd.read_excel(name)
        def replace(x):
            if x=='优秀':
                x=95
            if x=='良好':
                x=85
            if x=='及格':
                x==60
            return x
        df['成绩']=df['成绩'].map(replace)
        m2=np.array(df['成绩'])
        m1=np.array(df['学分'])
        s1=m1.sum()
        s2=np.dot(m1,m2.T)
        print(s2/s1)
        input('计算完成，按任意键结束。')
        ret=False
    except:
        print('读取文件失败，请输入正确的文件名。')