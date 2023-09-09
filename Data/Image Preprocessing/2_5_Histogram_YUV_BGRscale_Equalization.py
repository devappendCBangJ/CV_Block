# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import cv2
import numpy as np
import matplotlib.pylab as plt

# ==============================================================
# 1. 이미지 불러오기
# ==============================================================
# 1) 이미지 BGRScale로 읽기
img = cv2.imread('/media/hi/SK Gold P31/Capstone/GolfBall/9_2_COCO_FinalGolfBall_Crawling/images/train/golf ball in rough1_com.jpg')

# 2) BGR -> YUV
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
hist_original = cv2.calcHist([img_yuv[:,:,0]], [0], None, [256], [0, 256])

# ==============================================================
# 2. 이미지 정보 계산 + 시각화
# ==============================================================
# 1) YUVscale의 Y채널 (밝기 정보)에 Equalization 적용
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
hist_equal = cv2.calcHist([img_yuv[:,:,0]], [0], None, [256], [0, 256])

# 2) YUV -> BGR
img2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# 3) Histogram 시각화
hists = {'Original':hist_original, 'cv2.equalizeHist()':hist_equal}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,2,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()

# 4) 이미지 시각화
cv2.imshow('Before', img)
cv2.imshow('After', img2)
cv2.waitKey()
cv2.destroyAllWindows()