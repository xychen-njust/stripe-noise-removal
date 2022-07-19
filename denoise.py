import cv2
import numpy as np
from matplotlib import pyplot as plt
from function import hough

from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['FangSong']
# mpl.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(20, 20))

#读取图像
img_3 = cv2.imread('./test2.png', 1)
# img_3 = cv2.imread('./lena.png', 1)
img_3 = (img_3 - np.min(img_3)) / (np.max(img_3) - np.min(img_3)) * 255
img_3 = img_3.astype('uint8')
# print(img_3.shape)
noise = np.zeros_like(img_3)
for channel in range(3):
    img = img_3[:,:,channel]
    # print(img)
    #傅里叶变换
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 取对数为了得出mask
    fshift_log = np.log(1 + np.abs(fshift[:,:]))

    # 归一化二值化
    fshift_log = (fshift_log - np.min(fshift_log)) / (np.max(fshift_log) - np.min(fshift_log)) * 255
    # fshift_log = hough(fshift_log)
    # fshift_log = np.where(fshift_log[...] < 150, 0, 255)
    fshift_log = fshift_log.astype('uint8')
    # plt.imshow(fshift_log, 'gray')
    # plt.show()

    #去除中心部分
    # rows, cols,_ = fshift_log.shape
    # crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    # kuandu = 250
    # fshift_log[crow - kuandu:crow + kuandu, ccol -kuandu:ccol+kuandu] = 0
    # fshift_log = hough(fshift_log)

    # plt.imshow(fshift_log, 'gray')
    # plt.show()


    #设置梭形滤波器
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2) #中心位置
    mask = np.ones((rows, cols, 2))
    mask2 = np.zeros((rows, cols), np.uint8)
    # lidu = 2
    # kuandu = 10
    # mask[crow-kuandu:crow+kuandu, 0:ccol-lidu] = 0
    # mask[crow-kuandu:crow+kuandu, ccol+lidu:cols] = 0

    zero = 17
    # 条纹方向
    angle = 8
    kesi1 = np.tan((zero+angle)/180 * np.pi)
    # print(kesi1)
    kesi2 = np.tan((zero-angle) / 180 * np.pi)
    for size in range(ccol-2):
        size = size+2
        location = cols+size
        d1 = int(np.ceil(size*kesi1))
        d2 = int(np.floor(size * kesi2))

        m1 = np.where(fshift_log[crow + d2:crow + d1, ccol + size] >100, 0,1)
        m2 = np.where(fshift_log[crow - d1:crow - d2, ccol - size] >100, 0,1)

        fshift[crow + d2:crow + d1, ccol + size, :] =  fshift[crow + d2:crow + d1, ccol + size, :]*m1

        fshift[crow - d1:crow - d2, ccol - size, :] = fshift[crow - d1:crow - d2, ccol - size, :] *m2

    #掩膜图像和频谱图像乘积
    f = fshift * mask

    s_low=np.log(np.abs(f))
    #傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
    s_low=cv2.magnitude(s_low[:,:,0], s_low[:,:,1])

    plt.imshow(s_low, 'gray')
    plt.show()
    plt.imshow(res, 'gray')

    plt.show()
    res = (res-np.min(res))/(np.max(res)-np.min(res)) *255
    res = np.rint(res)
    print(res)
    # 结果
    noise[:,:,channel]=res


cv2.imwrite('./denoise.png', noise)


