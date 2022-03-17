import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#1번쨰 -> 이미지 창 openCV로 띄우기
img = cv.imread("home.jpg")
cv.imshow("test",img)
cv.waitKey(0)

#2번쨰 -> matplotlib 에서 subplot 사용하기
img2 = cv.imread("test_img.jpg")
# subplot() 은 현재 조작중인 figure 객체(인터페이스) 에 subplot을 만들고 조작하는 것.
# subplot(행의 수, 열의 수, 위치(1행1열부터 0,1,2 ... 1행 다세면 2행으로 ))
plt.subplot(1,4,1)
plt.imshow(img) #
plt.title("home")

plt.subplot(1,4,2)
plt.imshow(img2)
plt.title("test_img")

## plt.imshow() 는 이미지 삽입(image show). plt.show() 는 plot 창 띄우기
#plt.show() #-> plt.show()는 그래프 조작 후 맨 마지막에 1번만 출력해야함.

#3번째 -> convert Color : bgr -> rgb로!
img_bgr = cv.imread("home.jpg")
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

cv.imshow("rgb",img_rgb)
cv.waitKey(0)

img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
cv.imshow('GRAY',img_gray)
cv.waitKey(0)

# #4번째 -> Histogram 찍어보기
# # openCV 는 bgr 순서로 색을 받아들이기 때문에 channel 순서가 blue : 0, green : 1, red : 2로 할당된다.
# # calcHist 의 parameter 는 img파일, color channel, histsize(x축 범위 = 길이), range 순서이다.
# # histsize = 각 차원의 bin 개수
# # ranges = 각 차원의 분류 bin의 최소값 최대값을 의미한다.
# # 정리해보면 색은 0~255 수로 표현되고, histsize는 총 256개 존재할 수 있으며 최소값은 0, 최대값은 255 이다(=range)
hist_r = cv.calcHist(img_bgr,[2],None,[256],[0,255])
# hist_g = cv.calcHist(img_bgr,[1],None,[256],[0,255])
# hist_b = cv.calcHist(img_bgr,[0],None,[256],[0,255])


plt.subplot(1,4,3)
plt.plot(hist_r, color='r')
# plt.plot(hist_g, color='g')
# plt.plot(hist_b, color='b')
# plt.title("histogram")

# gray는 channel이 하나다.
hist_gray = cv.calcHist([img_gray], [0], None, [256], [0, 255])

plt.subplot(1,4,4)
plt.plot(hist_gray, color='black')
plt.show()

#5번째 궁금증 -> img 는 몇 픽셀인가?
#pirnt(img[0])