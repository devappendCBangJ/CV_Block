# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import glob
import shutil
import argparse

import numpy as np

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='2_YOLOv5_Only_Image_File_Split_Copy')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/BG-20k/train', type=str, help='Data Split할 폴더 경로 지정')
parser.add_argument('--split-path', default='_Split', type=str, help='Split 데이터셋을 저장할 폴더 경로 지정')
parser.add_argument('--source-parent-pathes', default=['images', 'labels'], type=str, nargs='*', help='source 폴더 기준 부모 폴더들 경로')
parser.add_argument('--source-child-pathes', default=['train', 'val', 'test'], type=str, nargs='*', help='source 폴더 기준 자식 폴더들 경로')
parser.add_argument('--train-size', default=2300, type=int, help='train data 개수')
parser.add_argument('--val-size', default=400, type=int, help='val data 개수')

args = parser.parse_args()

# ==============================================================
# 1. Data Split을 위한 폴더 생성
# ==============================================================
base_split_path = args.base_path + args.split_path
if not os.path.exists(base_split_path):
    for folder in args.source_parent_pathes:
        for split in args.source_child_pathes:
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

image_filenames = get_filenames(f'{args.base_path}')

# ==============================================================
# 3. Data Split
# ==============================================================
# --------------------------------------------------------------
# 1) list to numpy
# --------------------------------------------------------------
image_filenames = np.array(list(image_filenames))

# --------------------------------------------------------------
# 2) Data Shuffle
# --------------------------------------------------------------
np.random.seed(42)
np.random.shuffle(image_filenames)

# --------------------------------------------------------------
# 3) Data Split
# --------------------------------------------------------------
# (2) Data Split 함수 정의
def split_dataset(image_filenames, train_size, val_size):
    for idx, image_filename in enumerate(image_filenames):
        # 1] Data Split 기준
        if idx < train_size:
            split = 'train'
        elif idx < train_size + val_size:
            split = 'val'
        else:
            split = 'test'

        # 2] Image Source / Target 파일 경로 추출
        source_image_path = f'{args.base_path}/{image_filename}'
        target_image_path = f'{base_split_path}/images/{split}'

        # 3] Image 파일 복사
        shutil.copy(source_image_path, target_image_path)

# (1) Data Split
split_dataset(image_filenames, train_size=args.train_size, val_size=args.val_size)