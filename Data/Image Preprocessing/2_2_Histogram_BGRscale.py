# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import cv2
import matplotlib.pylab as plt

# ==============================================================
# 1. 이미지 불러오기
# ==============================================================
# 1) 이미지 BGRscale로 읽기
img = cv2.imread('/media/hi/SK Gold P31/Capstone/GolfBall/9_2_COCO_FinalGolfBall_Crawling/images/train/golf ball in rough1_com.jpg')

# ==============================================================
# 2. 이미지 정보 계산 + 시각화
# ==============================================================
# 1) 이미지 시각화
cv2.imshow('img', img)

# 2) Histogram 계산 + 정보 (histogram shape / histogram 총 합계 / image shape)
img_channels = cv2.split(img) # tuple 3 -> (image_height, image_width)
img_colors = ('b', 'g', 'r')
for (ch, color) in zip (img_channels, img_colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    print("hist.shape:", hist.shape) # hist.shape : (256,1)
    print("hist.sum():", hist.sum(), "ch.shape:",ch.shape) # hist.sum() : image_height x image_width # img.shape : (image_height, image_width)
    plt.plot(hist, color = color)

# 3) Histogram 시각화
plt.show()