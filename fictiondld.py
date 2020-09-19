import re
import os
from bs4 import BeautifulSoup
import requests
import time

def getfiction(url):
	try:
		r=requests.get(url,timeout=30)
		r.raise_for_status()
		r.encoding=r.apparent_encoding
		soup=BeautifulSoup(r.text,'html.parser')
		ul=soup.find_all('ul','list')[1]
		for a in ul('a'):
			fiction=str(a.get('href'))
			yield 'http://www.jinyongwang.com'+fiction
	except:
		yield ''

def getchpt(furl):
	try:
		r=requests.get(furl,timeout=30)
		r.raise_for_status()
		r.encoding=r.apparent_encoding
		soup=BeautifulSoup(r.text,'html.parser')
		ul=soup.find('ul','mlist')
		i=1
		n=len(ul('a'))
		start=time.perf_counter()
		for a in ul('a'):
			d='*'*i
			b='-'*(n-i)
			c=(i/n)*100
			dur=time.perf_counter()-start
			print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c,d,b,dur),end='')
			i=i+1			
			chpt=str(a.get('href'))
			yield 'http://www.jinyongwang.com'+chpt
	except:
		yield ''

def writetext(curl):
	try:
		r=requests.get(curl,timeout=30)
		r.raise_for_status()
		r.encoding=r.apparent_encoding
		soup=BeautifulSoup(r.text,'html.parser')
		title=soup.find('title')
		book=re.search(r'_.*_',str(title))
		bstr=book.group(0)[1:-1]
		chpt=re.search(r'第.*?_',str(title))
		cstr=chpt.group(0)[:-1]
		directory=os.getcwd()
		root=os.path.join(directory,bstr)
		path=os.path.join(root,cstr+'.txt')
		if not os.path.exists(root):
			os.mkdir(root)
		f=open(path,'wt',encoding="utf-8")
		cnt=0
		for p in soup('p'):
			if cnt<3:
				f.write('')
			else:
				f.write('  ')
				f.write(str(p.string)+'\n')
			cnt=cnt+1
	except:
		return ''

def main():
	url='http://www.jinyongwang.com/book/'
	n=0
	for furl in getfiction(url):
		if n%2==0:
			print('正在下载第{}本书'.format(int(n/2+1)))
			for curl in getchpt(furl):
				writetext(curl)
			print('\n第{}本书下载完成'.format(int(n/2+1)))
		n=n+1
	input('\r下载完成，按任意键结束')

main()