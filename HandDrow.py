from PIL import Image
import numpy as np

imgname = input('请输入模板图片名称：')
a = np.asarray(Image.open(imgname).convert('L')).astype('float')

depth = 10.
grad = np.gradient(a)
grad_x, grad_y = grad
grad_x = grad_x * depth / 100.
grad_y = grad_y * depth / 100.
A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
uni_x = grad_x / A
uni_y = grad_y / A
uni_z = 1. / A

vec_el = np.pi / 2.2
vec_az = np.pi / 4.
dx = np.cos(vec_el) * np.cos(vec_az)
dy = np.cos(vec_el) * np.sin(vec_az)
dz = np.sin(vec_el)

b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
b = b.clip(0, 255)

im = Image.fromarray(b.astype('uint8'))
end = imgname[-4:]
start = imgname[:-4] + 'HD'
name2 = start + end
im.save(name2)
input('程序执行完毕，请按任意键退出。')
