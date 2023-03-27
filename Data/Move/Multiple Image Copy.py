# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import glob
import shutil

import numpy as np
from PIL import Image

# ==============================================================
# 0. 변수 정의
# ==============================================================
base_path = './Dataset/'
from_path = ['Golf Ball Detection.v2i.yolov5pytorch', 'GolfBallDetector.v10i.yolov5pytorch', 'Golfwithme.v11-fixing.yolov5pytorch', 'Heimdallr.v1i.yolov5pytorch', 'FinalGolf.v6-betterannotationsall3.yolov5pytorch']
train_filenames = []
val_filenames = []
test_filenames = []

# ==============================================================
# 1. Data Split을 위한 폴더 생성
# ==============================================================
base_split_path = base_path + 'Golf_Ball_Split'
if not os.path.exists(base_split_path):
    for folder in ['images', 'labels']:
        for split in ['train', 'val', 'test']:
            os.makedirs(f'{base_split_path}/{folder}/{split}')

# ==============================================================
# 2. 폴더 내 파일명 중복 확인
# ==============================================================
# --------------------------------------------------------------
# 1) 각 클래스별 파일명 추출
# --------------------------------------------------------------
def get_filenames(folder_path):
    filenames = set()
    for file_path in glob.glob(os.path.join(folder_path, '*.jpg')):
        filenames.add(file_path)
    return filenames

for idx, f in enumerate(from_path):
    train_filenames.extend(list(get_filenames(f'{base_path + f}/train/images')))
    val_filenames.extend(list(get_filenames(f'{base_path + f}/valid/images')))
    test_filenames.extend(list(get_filenames(f'{base_path + f}/test/images')))
    print(test_filenames)

# ==============================================================
# 3. Data Split
# ==============================================================
for idx_fs, image_filenames in enumerate([train_filenames, val_filenames, test_filenames]):
    print(image_filenames)
    for idx_f, image_filename in enumerate(image_filenames):
        # 1] Label Path 정의
        print(image_filename)
        label_filename = image_filename.replace('.jpg', '.txt')
        label_filename = label_filename.replace('images', 'labels', 1)
        print(label_filename)

        # 2] train / val / test 구분
        if idx_fs == 0:
            temp_path = 'train'
        elif idx_fs == 1:
            temp_path = 'val'
        elif idx_fs == 2:
            temp_path = 'test'

        # 3] Label + Image Target 파일 경로 추출
        target_image_path = f'{base_split_path}/images/{temp_path}'
        target_label_path = f'{base_split_path}/labels/{temp_path}'

        # 4] Label + Image 파일 복사
        shutil.copy(image_filename, target_image_path)
        shutil.copy(label_filename, target_label_path)
