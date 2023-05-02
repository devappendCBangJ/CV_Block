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
base_path = './Dataset/Cat_Dog'

# ==============================================================
# 1. Data Split을 위한 폴더 생성
# ==============================================================
base_split_path = base_path + '_Split'
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
        filename = os.path.split(file_path)[-1]
        filenames.add(filename)
    return filenames

dog_filenames = get_filenames(f'{base_path}/dog/images')
cat_filenames = get_filenames(f'{base_path}/cat/images')

# --------------------------------------------------------------
# 2) 중복 파일 확인
# --------------------------------------------------------------
duplicates_filenames = dog_filenames  & cat_filenames

# --------------------------------------------------------------
# 3) 중복 파일 시각화
# --------------------------------------------------------------
print(f"duplicates : {duplicates_filenames}")

for filename in duplicates_filenames:
    for animal in ['cat', 'dog']:
        Image.open(f'{base_path}/{animal}/images/{filename}').show()

# --------------------------------------------------------------
# 4) 중복 파일 제거
# --------------------------------------------------------------
dog_filenames -= duplicates_filenames

print(f"dog_filenames_len : {len(dog_filenames)}")
print(f"cat_filenames_len : {len(cat_filenames)}")

# ==============================================================
# 3. Data Split
# ==============================================================
# --------------------------------------------------------------
# 1) list to numpy
# --------------------------------------------------------------
dog_filenames = np.array(list(dog_filenames))
cat_filenames = np.array(list(cat_filenames))

# --------------------------------------------------------------
# 2) Data Shuffle
# --------------------------------------------------------------
np.random.seed(42)
np.random.shuffle(dog_filenames)
np.random.shuffle(cat_filenames)

# --------------------------------------------------------------
# 3) Data Split
# --------------------------------------------------------------
# (1) Data Split 함수 정의
def split_dataset(animal, image_filenames, train_size, val_size):
    for idx, image_filename in enumerate(image_filenames):
        # 1] Label Path 정의
        label_filename = image_filename.replace('.jpg', '.txt')

        # 2] Data Split 기준
        if idx < train_size:
            split = 'train'
        elif idx < train_size + val_size:
            split = 'val'
        else:
            split = 'test'

        # 3] Label + Image Source 파일 경로 추출
        source_image_path = f'{base_path}/{animal}/images/{image_filename}'
        source_label_path = f'{base_path}/{animal}/darknet/{label_filename}'

        # 4] Label + Image Target 파일 경로 추출
        target_image_path = f'{base_split_path}/images/{split}'
        target_label_path = f'{base_split_path}/labels/{split}'

        # 5] Label + Image 파일 복사
        shutil.copy(source_image_path, target_image_path)
        shutil.copy(source_label_path, target_label_path)

# (2) Cat Data & Dog Data Split
split_dataset('cat', cat_filenames, train_size=400, val_size=50)
split_dataset('dog', dog_filenames, train_size=399, val_size=49) # reduce the number by 1 for each set due to three duplicates
