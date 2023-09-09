# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import cv2
import matplotlib.pylab as plt

# ==============================================================
# 1. 이미지 불러오기
# ==============================================================
# 1) 이미지 GrayScale로 읽기
img = cv2.imread('/media/hi/SK Gold P31/Capstone/GolfBall/9_2_COCO_FinalGolfBall_Crawling/images/train/golf ball in rough1_com.jpg', cv2.IMREAD_GRAYSCALE)

# ==============================================================
# 2. 이미지 정보 계산 + 시각화
# ==============================================================
# 1) 이미지 시각화
cv2.imshow('img', img)

# 2) Histogram 계산
hist = cv2.calcHist([img], [0], None, [256], [0,256])

# 3) Histogram 정보 (histogram shape / histogram 총 합계 / image shape) + 시각화
print("hist.shape:", hist.shape) # hist.shape : (256,1)
print("hist.sum():", hist.sum(), "img.shape:",img.shape) # hist.sum() : image_height x image_width # img.shape : (image_height, image_width)
plt.plot(hist)
plt.show()