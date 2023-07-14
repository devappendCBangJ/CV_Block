# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import PIL
import argparse
import numpy as np
import tqdm

# --------------------------------------------------------------
# 2) 딥러닝 라이브러리 불러오기
# --------------------------------------------------------------
import torchvision.transforms as T

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='1_')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel_Remove_BboxInBbox_Remove_ErrorBboxRatio', type=str, help='라벨과 이미지 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-pathes', default=['images'], type=str, nargs='*', help='source 폴더 기준 부모 폴더들 경로')
parser.add_argument('--source-child-pathes', default=['train', 'val'], type=str, nargs='*', help='source 폴더 기준 자식 폴더들 경로')

args = parser.parse_args()

# ==============================================================
# 0. 함수 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 이미지의 RGB 채널별 통계량 확인
# --------------------------------------------------------------
def calculate_feature(image_filenames, normalize_val = None):
    # (1) 변수 초기화
    mean_r, mean_g, mean_b = 0, 0, 0
    std_r, std_g, std_b = 0, 0, 0
    min_r, min_g, min_b = 1000000000, 1000000000, 1000000000
    max_r, max_g, max_b = -1000000000, -1000000000, -1000000000

    # (2) 이미지 파일 불러오기 + RGB 채널별 통계량(mean, std, min, max) 계산
    tbar = tqdm.tqdm(image_filenames)
    for i, image_filename in enumerate(tbar):
        pil_img = PIL.Image.open(image_filename)
        resized_img = T.Resize(800)(pil_img)
        tensor_img = T.ToTensor()(resized_img)
        if normalize_val:
            tensor_img = T.Normalize(*normalize_val)(tensor_img)
        tensor_img = tensor_img.numpy()

        # 1] dataset의 axis=1, 2에 대한 평균 산출
        mean_ = np.mean(tensor_img, axis=(1, 2))
        # 2] r, g, b 채널에 대한 각각의 평균 산출
        mean_r += mean_[0]
        mean_g += mean_[1]
        mean_b += mean_[2]
        # 3] dataset의 axis=1, 2에 대한 표준편차 산출
        std_ = np.std(tensor_img, axis=(1, 2))
        # 4] r, g, b 채널에 대한 각각의 표준편차 산출
        std_r += std_[0]
        std_g += std_[1]
        std_b += std_[2]

        # 5] dataset의 axis=1, 2에 대한 최소값 산출
        min_ = np.min(tensor_img, axis=(1, 2))
        # 6] r, g, b 채널에 대한 각각의 최소값 산출
        if min_[0] < min_r:
            min_r = min_[0]
        if min_[1] < min_g:
            min_g = min_[1]
        if min_[2] < min_b:
            min_b = min_[2]

        # 7] dataset의 axis=1, 2에 대한 최대값 산출
        max_ = np.max(tensor_img, axis=(1, 2))
        # 8] r, g, b 채널에 대한 각각의 최대값 산출
        if max_[0] > max_r:
            max_r = max_[0]
        if max_[1] > max_g:
            max_g = max_[1]
        if max_[2] > max_b:
            max_b = max_[2]

    # (3) 통계량 최종 계산
    len_image_filenames = len(image_filenames)
    mean_r, mean_g, mean_b = mean_r/len_image_filenames, mean_g/len_image_filenames, mean_b/len_image_filenames
    std_r, std_g, std_b = std_r/len_image_filenames, std_g/len_image_filenames, std_b/len_image_filenames

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b), (min_r, min_g, min_b), (max_r, max_g, max_b)

# ==============================================================
# 1. Main문
# ==============================================================
# --------------------------------------------------------------
# 1) 데이터셋 불러오기
# --------------------------------------------------------------
# (1) Dataset Path에서 Image List 불러오기
image_filenames = []
for source_parent_path in args.source_parent_pathes:
    for source_child_path in args.source_child_pathes:
        dataset_path = f'{args.base_path}/{source_parent_path}/{source_child_path}'
        image_filenames.extend(f'{dataset_path}/{image_filename}' for image_filename in os.listdir(dataset_path))
print(f'the number of images : {len(image_filenames)}')

# --------------------------------------------------------------
# 2) 데이터의 평균, 표준편차 계산
# --------------------------------------------------------------
mean_, std_, min_, max_ = calculate_feature(image_filenames)
print('==='*10)
print(f'mean(R,G,B): {mean_}')
print(f'std(R,G,B): {std_}')
print(f'min(R,G,B): {min_}')
print(f'max(R,G,B): {max_}')
print('==='*10)

# --------------------------------------------------------------
# 3) 정규화 후 데이터의 평균, 표준편차 계산
# --------------------------------------------------------------
mean_, std_, min_, max_ = calculate_feature(image_filenames, (mean_, std_))
print('==='*10)
print(f'mean(R,G,B): {mean_}')
print(f'std(R,G,B): {std_}')
print(f'min(R,G,B): {min_}')
print(f'max(R,G,B): {max_}')
print('==='*10)
