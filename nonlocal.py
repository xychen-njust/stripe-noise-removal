import cv2
import numpy as np
from matplotlib import pyplot as plt
# 将图像分成多个patch来求傅里叶变换，由于条纹噪声存在于每一个patch中
img_3 = cv2.imread('./test/test.jpg', 1)
# img_3 = cv2.imread('./lena.png', 1)
img_3 = (img_3 - np.min(img_3)) / (np.max(img_3) - np.min(img_3)) * 255
img_3 = img_3.astype('uint8')
m,n,c= img_3.shape
img_3 = img_3[:,:,0]
# plt.imshow(img_3[:,:], 'gray')
# plt.show()
# img_3=cv2.GaussianBlur(img_3,(5,5),0)
# plt.imshow(img_3[:,:], 'gray')
# plt.show()
N = 100
strip = 8
k_x = (m-N)//strip
k_y = (n-N)//strip
print((k_x,k_y))
patchs = np.zeros((N,N,2))
for i in range(k_x):
    i1= i*strip
    for j in range(k_y):
        j1= j*strip
        patch =img_3[i1:i1+N,j1:j1+N]
        # 对每一个图像求频域
        dft = cv2.dft(np.float32(patch), flags=cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(dft)
        fshift_log = np.log(1 + np.abs(fshift[:, :]))
        fshift_log = (fshift_log - np.min(fshift_log)) / (np.max(fshift_log) - np.min(fshift_log)) * 255
        patchs+=fshift_log
patchs = patchs/(k_y*k_x)
s_low = cv2.magnitude(patchs[:, :, 0], patchs[:, :, 1])
plt.imshow(s_low[:,:], 'gray')
plt.show()
