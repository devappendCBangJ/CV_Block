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
# 1) YUVscale의 Y채널 (밝기 정보)에 Equalization 적용 + BGR -> YUV
img_eq = img_yuv.copy()
img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
hist_equal = cv2.calcHist([img_eq[:,:,0]], [0], None, [256], [0, 256])
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

# 2) YUVscale의 Y채널 (밝기 정보)에 CLAHE 적용 + BGR -> YUV
img_clahe = img_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # CLAHE 생성
img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])            # CLAHE 적용
hist_clahe = cv2.calcHist([img_clahe[:,:,0]], [0], None, [256], [0, 256])
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

# 3) Histogram 시각화
hists = {'Original':hist_original, 'cv2.equalizeHist()':hist_equal, 'cv2.createCLAHE()':hist_clahe}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()

# 4) 이미지 시각화
cv2.imshow('Original', img)
cv2.imshow('EqualizationHist', img_eq)
cv2.imshow('CLAHE', img_clahe)
cv2.waitKey()
cv2.destroyAllWindows()