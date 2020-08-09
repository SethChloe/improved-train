#图像的加减乘除运算
import numpy as np 
from PIL import Image

class ImageObject(object):
	"""docstring for ImageObject"""
	def __init__(self, path=""):
		super(ImageObject, self).__init__()
		self.path = path
		try:
			self.data=np.array(Image.open(path))
		except:
			self.data=None

	def __add__(self,other):
		image=ImageObject()
		try:
			image.data=np.mod(self.data+other.data, 255)
		except:
			image.data=self.data
		return image

	def __sub__(self,other):
		image=ImageObject()
		try:
			image.data=np.mod(self.data-other.data, 255)
		except:
			image.data=self.data
		return image

	def __mul__(self,factor):
		image=ImageObject()
		try:
			image.data=np.mod(self.data*factor, 255)
		except:
			image.data=self.data
		return image

	def __truediv__(self,factor):
		image=ImageObject()
		try:
			image.data=np.mod(self.data//factor, 255)
		except:
			image.data=self.data
		return image

	def saveImage(self,path):
		try:
			im=Image.fromarray(self.data)
			im.save(path)
			return True
		except:
			return False

x=input("请输入你想要执行的图像操作:(+,-,*,/)")
if x=="+":
	patha=input("请输入第一个图像的文件名:")
	pathb=input("请输入第二个图像的文件名(两图像的尺寸需相同):")
	a=ImageObject(patha)
	b=ImageObject(pathb)
	(a+b).saveImage("result_add.png")
elif x=="-":
	patha=input("请输入第一个图像的文件名:")
	pathb=input("请输入第二个图像的文件名(两图像的尺寸需相同):")
	a=ImageObject(patha)
	b=ImageObject(pathb)
	(a-b).saveImage("result_sub.png")	
elif x=="*":
	patha=input("请输入图像的文件名:")
	pathb=input("请输入乘数:")
	a=ImageObject(patha)
	b=eval(pathb)
	(a*b).saveImage("result_mul.png")
else:
	patha=input("请输入图像的文件名:")
	pathb=input("请输入除数:")
	a=ImageObject(patha)
	b=eval(pathb)
	(a/b).saveImage("result_truediv.png")

input("操作成功，按任意键结束。")
