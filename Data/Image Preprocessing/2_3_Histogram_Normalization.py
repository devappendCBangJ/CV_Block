# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import cv2
import numpy as np
import matplotlib.pylab as plt

# ==============================================================
# 1. 이미지 불러오기
# ==============================================================
# 1) 이미지 GrayScale로 읽기
img = cv2.imread('/media/hi/SK Gold P31/Capstone/GolfBall/9_2_COCO_FinalGolfBall_Crawling/images/train/golf ball in rough1_com.jpg', cv2.IMREAD_GRAYSCALE)

# ==============================================================
# 2. 이미지 정보 계산 + 시각화
# ==============================================================
# 1) 정규화
# (1way) 직접 정규화
img_f = img.astype(np.float32) # img_f.shape : (image_height, image_width)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min())) # img_norm.shape : (image_height, image_width)
img_norm = img_norm.astype(np.uint8) # img_norm.shape : (image_height, image_width)

# (2way) OpenCV 함수 정규화
img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) # img_norm2.shape : (image_height, image_width)

# 2) 이미지 시각화
cv2.imshow('Before', img)
cv2.imshow('Manual', img_norm)
cv2.imshow('cv2.normalize()', img_norm2)

# 3) Histogram 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

# 4) Histogram 시각화
hists = {'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()