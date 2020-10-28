from IPython.display import display,Latex,Math
import numpy as np
# %matplotlib inline
 
from IPython.core.interactiveshell import InteractiveShell
sh = InteractiveShell.instance()
 
def number_to_str(n,cut=5):
    ns=str(n)
    format_='{0:.'+str(cut)+'f}'
    if 'e' in ns or ('.' in ns and len(ns)>cut+1):
        return format_.format(n)
    else:
        return str(n)
 
def matrix_to_latex(mat,style='bmatrix'):
    if type(mat)==np.matrixlib.defmatrix.matrix:
        mat=mat.A
    head=r'\begin{'+style+'}'
    tail=r'\end{'+style+'}'
    if len(mat.shape)==1:
        body=r'\\'.join([str(el) for el in mat])
        return head+body+tail
    elif len(mat.shape)==2:
        lines=[]
        for row in mat:
            lines.append('&'.join([number_to_str(el)  for el in row])+r'\\')
        s=head+' '.join(lines)+tail
        return s
    return None

sh.display_formatter.formatters['text/latex'].type_printers[np.ndarray]=matrix_to_latex

def save(name,arr,f):
    np.savetxt(name+'.csv',arr,fmt='%'+f,delimiter=',')