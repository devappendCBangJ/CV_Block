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
parser = argparse.ArgumentParser(description='2_YOLOv5_Only_ImageFile_Split_Copy')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/GSC/Sentiment_Analysis/Korean_Emotion_Movie/original/val', type=str, help='Data Split할 이미지의 grandmother 경로 지정')
parser.add_argument('--source-parent-pathes', default=['슬픔', '당황', '기쁨', '불안', '상처'], type=str, nargs='*', help='Data Split할 이미지의 mother 경로들 지정')
parser.add_argument('--base-split-path', default='/media/hi/SK Gold P31/GSC/Sentiment_Analysis/Korean_Emotion_Movie/original/val/Split', type=str, help='Split 데이터셋을 저장할 폴더 경로 지정')
parser.add_argument('--train-size', default=1750, type=int, help='train data 개수')
parser.add_argument('--val-size', default=250, type=int, help='val data 개수')
parser.add_argument('--test-size', default=0, type=int, help='test data 개수')

args = parser.parse_args()

# ==============================================================
# 0. 함수 정의
# ==============================================================
# --------------------------------------------------------------
# 1) Data Split 함수 정의
# --------------------------------------------------------------
def split_dataset(source_parent_abs_path, image_filenames, train_size, val_size, test_size):
    for idx, image_filename in enumerate(image_filenames):
        # 1] Data Split 기준
        if idx < train_size:
            split = 'train'
        elif idx < train_size + val_size:
            split = 'val'
        elif idx < train_size + val_size + test_size:
            split = 'test'
        else:
            break

        # 2] Image Source / Target 파일 경로 추출
        source_image_path = f'{source_parent_abs_path}/{image_filename}'
        target_image_path = f'{args.base_split_path}/images/{split}'

        # 3] Image 파일 복사
        shutil.copy(source_image_path, target_image_path)

        # 4] 시각화
        print(f'{source_parent_abs_path} Data Split {idx}/{len(image_filenames)} 완료')

# --------------------------------------------------------------
# 2) 파일명 추출 함수
# --------------------------------------------------------------
def get_filenames(folder_path):
    filenames = set()
    for file_path in glob.glob(os.path.join(folder_path, '*.jpg')):
        filename = os.path.split(file_path)[-1]
        filenames.add(filename)
    return filenames

# ==============================================================
# 1. Data Split을 위한 폴더 생성
# ==============================================================
# --------------------------------------------------------------
# 1) Data Split을 위한 폴더 생성
# --------------------------------------------------------------
if not os.path.exists(args.base_split_path):
    for folder in args.source_parent_pathes:
        for split in args.source_child_pathes:
            os.makedirs(f'{args.base_split_path}/{folder}/{split}')

for source_parent_path in args.source_parent_pathes:
    # --------------------------------------------------------------
    # 2) 각 클래스별 파일명 추출
    # --------------------------------------------------------------
    source_parent_abs_path = f'{args.base_path}/{source_parent_path}'
    image_filenames = get_filenames(f'{source_parent_abs_path}')

    # --------------------------------------------------------------
    # 3) Data Split
    # --------------------------------------------------------------
    # (1) list to numpy
    image_filenames = np.array(list(image_filenames))

    # (2) Data Shuffle
    np.random.seed(42)
    np.random.shuffle(image_filenames)

    # (3) Data Split
    split_dataset(source_parent_abs_path, image_filenames, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size)