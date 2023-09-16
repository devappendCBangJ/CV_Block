# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import cv2
import math
import numpy as np
from PIL import Image

# ==============================================================
# 0. 함수 정의
# ==============================================================
def Custom_getGaussianKernel(kernel_size, sigma):
    """
    1D 가우시안 필터를 생성한다.
    :param size: int 커널 사이즈
    :param sigma: float
    :return kernel: np.array
    """
    assert kernel_size % 2 == 1, "Filter Dimension must be odd"                             # kernel_size : 무조건 홀수
    arr = np.arange(math.trunc(kernel_size / 2) * (-1), math.ceil(kernel_size / 2) + 1, 1)  # arr : [0 - kernel_size // 2 ~ 0 + kernel_size // 2 + 1]
    raw_gaussian_kernel = np.exp((-arr**2) / (2 * sigma**2))                                # 가우시안 필터 공식
    normalized_gaussian_kernel = raw_gaussian_kernel / raw_gaussian_kernel.sum()            # 가우시안 필터 정규화
    return normalized_gaussian_kernel

print("GaussianKernel Test : ", Custom_getGaussianKernel(5, 3))

# ==============================================================
# 1. 가우시안 필터 적용
# ==============================================================
# 1) 1D 가우시안 필터 생성
kernel_1d = cv2.getGaussianKernel(5, 3) # kernel_size : 5, sigma : 3
"""
kernel_1d.shape : (5, 1)
kernel_1d : [[0.17820326], [0.21052227], [0.22254894], [0.21052227], [0.17820326]]
"""

# 2) 2D 가우시안 필터 생성
kernel_2d = np.outer(kernel_1d, kernel_1d.transpose()) # (5, 1) x (1, 5)
"""
kernel_2d.shape : (5, 5)
kernel_2d : [[0.0317564  0.03751576 0.03965895 0.03751576 0.0317564 ], [0.03751576 0.04431963 0.04685151 0.04431963 0.03751576], [0.03965895 0.04685151 0.04952803 0.04685151 0.03965895], [0.03751576 0.04431963 0.04685151 0.04431963 0.03751576], [0.0317564  0.03751576 0.03965895 0.03751576 0.0317564 ]]
"""

# 3) 이미지 불러오기
img = Image.open("newskin.jpeg")
img_array = np.array(img)

# 4) 2D 가우시안 필터 적용 - 저주파 영역만 남김
low_img_array = cv2.filter2D(img_array, -1, kernel_2d)
low_img = Image.fromarray(low_img_array)
low_img.save('low_newskin.jpeg', 'bmp')

# 5) 2D 가우시안 필터값을 뺌 - 고주파 영역만 남김
high_img_array = img_array - low_img_array + 128
high_img = Image.fromarray(high_img_array)
high_img.save('high_newskin.jpeg', 'bmp')