# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import shutil
import argparse

import numpy as np

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='1_2_YOLOv5_Image_Label_TrainValTest_Split')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar', type=str, help='Data Split할 데이터들이 모여있는 base 경로 지정')
parser.add_argument('--base-image-path', default='images/train', type=str, help='Data Split할 base 경로 아래의 image 경로 지정')
parser.add_argument('--base-label-path', default='labels/train', type=str, help='Data Split할 base 경로 아래의 label 경로 지정')
parser.add_argument('--split-path', default='_Split', type=str, help='Split 데이터셋을 저장할 폴더 경로 지정')
parser.add_argument('--source-parent-pathes', default=['images', 'labels'], type=str, nargs='*', help='source 폴더 기준 부모 폴더들 경로')
parser.add_argument('--source-child-pathes', default=['train', 'val', 'test'], type=str, nargs='*', help='source 폴더 기준 자식 폴더들 경로')

parser.add_argument('--before-file-extension', default='.jpg', type=str, help='base_path/label_folder/source_child_folders 안에 들어있는 텍스트 파일 확장자')
parser.add_argument('--after-file-extension', default='.txt', type=str, help='base_path/after_folder/source_child_folders 에서 오픈할 이미지 파일 확장자')

parser.add_argument('--train-ratio', default=0.8, type=int, help='train data 비율')
parser.add_argument('--val-ratio', default=0.2, type=int, help='val data 비율')

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
    filenames = os.listdir(folder_path)
    return filenames

image_filenames = get_filenames(f'{args.base_path}/{args.base_image_path}')

# ==============================================================
# 3. Data Split
# ==============================================================
# --------------------------------------------------------------
# 1) list to numpy
# --------------------------------------------------------------
image_filenames = np.array(image_filenames)

# --------------------------------------------------------------
# 2) Data Shuffle
# --------------------------------------------------------------
np.random.seed(42)
np.random.shuffle(image_filenames)

# --------------------------------------------------------------
# 3) Data Split
# --------------------------------------------------------------
# (1) Data Split 함수 정의
def split_dataset(image_filenames, train_ratio, val_ratio):
    print(f"len(image_filenames) : {len(image_filenames)}")
    for idx, image_filename in enumerate(image_filenames):
        # 1] Label 파일명 추출
        label_filename = image_filename.replace(args.before_file_extension, args.after_file_extension)

        # 2] Image 비율
        image_ratio = idx / len(image_filenames)

        # 3] Data Split 기준
        if image_ratio <= train_ratio:
            split_folder = 'train'
        elif image_ratio <= train_ratio + val_ratio:
            split_folder = 'val'
        else:
            split_folder = 'test'

        # 4] Source / Target 파일 경로 추출
        source_image_path = f'{args.base_path}/{args.base_image_path}/{image_filename}'
        target_image_path = f'{base_split_path}/images/{split_folder}/{image_filename}'

        source_label_path = f'{args.base_path}/{args.base_label_path}/{label_filename}'
        target_label_path = f'{base_split_path}/labels/{split_folder}/{label_filename}'

        # 5] Image 파일 복사
        shutil.copy(source_image_path, target_image_path)
        shutil.copy(source_label_path, target_label_path)

        print(f'target_image_path : ', target_image_path)

split_dataset(image_filenames, train_ratio=args.train_ratio, val_ratio=args.val_ratio)