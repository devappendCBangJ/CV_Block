# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import numpy as np
import cv2

# ==============================================================
# 0. 함수 정의
# ==============================================================
# (2) Single Scale Retinex : log(반사성분) = log(입력성분) - log(배경성분)
def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance)) # (0, 0) : 3번째 인자인 variance에서 계산한 커널의 크기를 자동으로 결정. variance가 클수록 커널의 크기가 커짐!!!
    return retinex

# (2) Multi Scale Retinex : 여러 Scale의 Variance에 대한 [log(반사성분) = log(입력성분) - log(배경성분)] 가중합
def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance) # (0, 0) : 3번째 인자인 variance에서 계산한 커널의 크기를 자동으로 결정. variance가 클수록 커널의 크기가 커짐!!!
    retinex = retinex / len(variance_list)
    return retinex

# --------------------------------------------------------------
# 4) Multi Scale Retinex
# --------------------------------------------------------------
def MSR(img, variance_list):
    # (1) img 모든 값에 + 1 (log를 씌울 때 0이 들어가서 -무한대가 되는 경우를 방지)
    img = np.float64(img) + 1.0

    # (2) Multi Scale Retinex : 여러 Scale의 Variance에 대한 [log(반사성분) = log(입력성분) - log(배경성분)] 평균
    img_retinex = multiScaleRetinex(img, variance_list)

    # (3) RGB 각각의 채널에 대해서
    for i in range(img_retinex.shape[2]):
        # 1] 모든 반사 성분 결과값에 x100 -> 정수 변환
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)

        # 2] 모든 반사 성분에 100을 곱한 픽셀값 중에서 정수 변환시켰을 때, 0인 값의 개수 (x100 : 임의 지정. 소수점 연상의 수치적 불안정성 방지 + 연삭 속도 향상 + 히스토그램 구할 수 있는 장점!!!)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        # 3] 반사 성분의 기존 스케일에서 최대, 최소값 초기화
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0

        # 4] 조건에 따라 반사 성분 기존 스케일의 최대, 최소값 업데이트
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1: # 특정 반사 성분이 음수 + 0인 값의 개수와 비교했을 때, 10% 미만인 경우 업데이트 (개수가 적은거랑 뭔상관이지?)
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1: # 특정 반사 성분이 양수 + 0인 값의 개수와 비교했을 때, 10% 미만인 경우 업데이트 (개수가 적은거랑 뭔상관이지?)
                high_val = u / 100.0
                break

        # 5] 모든 반사 성분의 픽셀값 제한
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        # # 6] 모든 반사 성분의 픽셀값 [0 ~ 255] 정규화
        # img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) * 255

    # 6] 모든 반사 성분의 픽셀값 [0 ~ 255] 정규화
    overall_min = np.min(img_retinex)
    overall_max = np.max(img_retinex)
    img_retinex = (img_retinex - overall_min) / (overall_max - overall_min) * 255

    # 7] 모든 반사 성분의 최종 결과값 -> uint8 변환
    img_retinex = np.uint8(img_retinex)
    return img_retinex

# --------------------------------------------------------------
# 3) Single Scale Retinex
# --------------------------------------------------------------
def SSR(img, variance):
    # (1) img 모든 값에 + 1 (log를 씌울 때 0이 들어가서 -무한대가 되는 경우를 방지)
    img = np.float64(img) + 1.0

    # (2) Single Scale Retinex : log(반사성분) = log(입력성분) - log(배경성분)
    img_retinex = singleScaleRetinex(img, variance)

    # (3) RGB 각각의 채널에 대해서
    for i in range(img_retinex.shape[2]):
        # 1] 모든 반사 성분 결과값에 x100 -> 정수 변환
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)

        # 2] 모든 반사 성분에 100을 곱한 픽셀값중에서 정수 변환시켰을 때, 0인 값의 개수 (x100 : 임의 지정. 소수점 연상의 수치적 불안정성 방지 + 연삭 속도 향상 + 히스토그램 구할 수 있는 장점!!!)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        # 3] 반사 성분의 최대, 최소값 초기화
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0

        # 4] 조건에 따라 반사 성분의 최대, 최소값 업데이트
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        # 5] 모든 반사 성분의 픽셀값 제한
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        # 6] 모든 반사 성분의 픽셀값 [0 ~ 255] 정규화
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) * 255

    # 7] 모든 반사 성분의 최종 결과값 -> uint8 변환
    img_retinex = np.uint8(img_retinex)
    return img_retinex

# ==============================================================
# 1. Retinex
# ==============================================================
# 1) Gaussian Filter의 Variance 정의
variance_list=[500, 100] # 작은 Scale : 정규 분포 폭이 좁음 = 작은 영역의 조명 변화에 민감 = 커널의 크기가 작음, 큰 Scale : 정규 분포 폭이 넓음 = 큰 영역의 조명 변화에 민감 = 커널의 크기가 큼!!!
variance=100

# 2) 이미지 불러오기
img = cv2.imread('golf ball in rough3.jpg')

# 3) Single Scale Retinex
img_ssr = SSR(img, variance)

# 4) Multi Scale Retinex
img_msr = MSR(img, variance_list)

# 5) 이미지 시각화
cv2.imshow('Original', img)
cv2.imshow('SSR', img_ssr)
cv2.imshow('MSR', img_msr)
cv2.imwrite('SSR.jpg', img_ssr)
cv2.imwrite('MSR.jpg',img_msr)

cv2.waitKey(0)
cv2.destroyAllWindows()