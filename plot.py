from pylab import *
import numpy as np
from matplotlib.pyplot import MultipleLocator

data_path = r'val_results/'
ori_path = data_path + 'ori/'; v1_path = data_path + 'v1/'; v2_path = data_path + 'v2/'; v4_path = data_path + 'v4/'
v5_1_path = data_path + 'v5(1)/'; v6_path = data_path + 'v6/'
ori_img1 = np.load(ori_path + 'img1_val.npy'); ori_img2 = np.load(ori_path + 'img2_val.npy')
v1_img1 = np.load(v1_path + 'img1_val.npy'); v1_img2 = np.load(v1_path + 'img2_val.npy')
v2_img1 = np.load(v2_path + 'img1_val.npy'); v2_img2 = np.load(v2_path + 'img2_val.npy')
v4_img1 = np.load(v4_path + 'img1_val.npy'); v4_img2 = np.load(v4_path + 'img2_val.npy')
v5_1_img1 = np.load(v5_1_path + 'img1_val.npy'); v5_1_img2 = np.load(v5_1_path + 'img2_val.npy')
v6_img1 = np.load(v6_path + 'img1_val.npy'); v6_img2 = np.load(v6_path + 'img2_val.npy')

x = [iter for iter in range(1000,200000,1000)]

plt.figure( figsize=(20,8), dpi=180)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
plt.rcParams['axes.unicode_minus'] = False 
x_major_locator=MultipleLocator(20000) #把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.5) #把y轴的刻度间隔设置为10，并存在变量里
# x_major_locator=MultipleLocator(5) #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(50) #把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca() #ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator) #把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator) #把y轴的主刻度设置为10的倍数
plt.xlim(0,200000)
plt.ylim(30, 37) # T2: 37; PD: 38
plt.plot(x, ori_img1, 'ro-', color='#000000', alpha=0.8, marker=None, linewidth=1, label='MC-CDic')
# plt.plot(x[0:60], v1_img1, 'ro-', color='#33CCFF', alpha=0.8, marker=None, linewidth=1, label='Update-UV')
plt.plot(x[0:79], v2_img1, 'ro-', color='#01605F', alpha=0.8, marker=None, linewidth=1, label='Update-UCV')
plt.plot(x, v4_img1, 'ro-', color='#e60039', alpha=0.8, marker=None, linewidth=1, label='Delete FMF')
plt.plot(x, v5_1_img1, 'ro-', color='#4d1f00', alpha=0.8, marker=None, linewidth=1, label='Reconstruct x_2')
plt.plot(x[0:109], v6_img1, 'ro-', color='#33CCFF', alpha=0.8, marker=None, linewidth=1, label='Update-UCV*')

# plt.plot(Imax_terms, rate_ours, 'ro-', color='#e60039', alpha=0.8, marker='o', linewidth=1, label='EXNet-1')
# plt.plot(Imax_terms, rate_ours_new, 'ro-', color='#003153', alpha=0.8, marker=None, linewidth=1, label='EXNet-3')
# plt.plot(Imax_terms, power, 'ro-', color='#33CCFF', alpha=0.8, marker=None, linewidth=1, label='power')
# plt.plot(Imax_terms, rate, 'ro-', color='#4d1f00', alpha=0.8, marker=None, linewidth=1, label='W_diff')
# plt.plot(Imax_terms, rate, 'ro-', color='#003153', alpha=0.8, marker=None, linewidth=1, label='real')

plt.grid(True, linestyle='--', alpha=0.5) #默认是True，风格设置为虚线，alpha为透明度
plt.legend(loc="upper right")
plt.xlabel('iteration')
plt.ylabel('PSNR')
plt.title('T2 val_PSNR in IXI dataset')
plt.show()