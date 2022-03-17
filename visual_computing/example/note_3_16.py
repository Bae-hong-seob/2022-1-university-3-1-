import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_bgr = cv.imread("home.jpg")
cv.imshow('test', img_bgr)

img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# 보통 rgb인데 gbr을 염두해서 뒤에서부터 channel카운팅한다
hist_r = cv.calcHist([img_bgr], [0], None, [256], [0, 255])
hist_g = cv.calcHist([img_bgr], [1], None, [256], [0, 255])
hist_b = cv.calcHist([img_bgr], [2], None, [256], [0, 255])

hist_gray = cv.calcHist([img_gray], [0], None, [256], [0, 255])

plt.subplot(1, 3, 1)
plt.imshow(img_bgr)
plt.title('plt1')
plt.subplot(1, 3, 2)
plt.plot(hist_r, color='r')
plt.plot(hist_g, color='g')
plt.title('plt2')

plt.subplot(1, 3, 3)
plt.plot(hist_gray, color='black')
plt.title('plt3')

plt.show()


cv.waitKey(0)