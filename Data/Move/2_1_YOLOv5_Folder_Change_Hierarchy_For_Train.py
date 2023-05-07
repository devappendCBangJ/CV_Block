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

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='2_1_YOLOv5_Folder_Hierarchy_For_Train')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/Golfball_Near', type=str, help='옮길 데이터셋 폴더들이 존재하는 부모 폴더 경로 지정')
parser.add_argument('--move-path', default='_Move', type=str, help='데이터셋을 옮길 폴더 경로 지정')
parser.add_argument('--class-pathes', default=['CellPhoneV1_Move', 'CellPhoneV2_Move', 'MobilePhoneV1_Move', 'MobilePhoneV5_Move', 'PhoneDetectionV1_Move', 'PhoneDetectionV5_Move'], type=str, nargs='*', help='옮길 데이터셋 폴더명들 지정')
parser.add_argument('--target-parent-pathes', default=['images', 'labels'], type=str, nargs='*', help='target 기준 부모 폴더들 경로')
parser.add_argument('--target-child-pathes', default=['train', 'val', 'test'], type=str, nargs='*', help='target 기준 자식 폴더들 경로')

args = parser.parse_args()

# ==============================================================
# 1. 파일 이동을 위한 폴더 생성
# ==============================================================
base_all_path = args.base_path + args.move_path
if not os.path.exists(base_all_path):
    for folder in args.target_parent_pathes:
        for split in args.target_child_pathes:
            os.makedirs(f'{base_all_path}/{folder}/{split}')

# ==============================================================
# 2. 파일 계층 이동
# ==============================================================
# --------------------------------------------------------------
# 1) 각 파일명 추출
# --------------------------------------------------------------
def get_filenames(folder_path, image_path):
    filenames = set()
    if image_path == 'images':
        temp_path = '*.jpg'
    elif image_path == 'labels':
        temp_path = '*.txt'
    for file_path in glob.glob(os.path.join(folder_path, temp_path)):
        filename = os.path.split(file_path)[-1]
        filenames.add(filename)
    return filenames

# --------------------------------------------------------------
# 2) 파일 계층 이동 (base_path -> train_path -> image_path 순회하면서 base_all_path -> image_path -> train_path으로 합침)
# --------------------------------------------------------------
for train_path in args.target_child_pathes:
    for image_path in args.target_parent_pathes:
        source_filenames = get_filenames(f'{args.base_path}/{train_path}/{image_path}', image_path)
        target_folder_path = f'{base_all_path}/{image_path}/{train_path}'
        for source_filename in source_filenames:
            source_file_path = f'{args.base_path}/{train_path}/{image_path}/{source_filename}'
            print(source_file_path)
            print(target_folder_path)
            shutil.move(source_file_path, target_folder_path)

# --------------------------------------------------------------
# 3) Source에서 파일명이 valid인 경우 val로 계층 이동하는 경우 따로 처리
# --------------------------------------------------------------
source_train_path = 'valid'
target_train_path = 'val'
for image_path in args.target_parent_pathes:
    source_filenames = get_filenames(f'{args.base_path}/{source_train_path}/{image_path}', image_path)
    target_folder_path = f'{base_all_path}/{image_path}/{target_train_path}'
    for source_filename in source_filenames:
        source_file_path = f'{args.base_path}/{source_train_path}/{image_path}/{source_filename}'
        print(source_file_path)
        print(target_folder_path)
        shutil.move(source_file_path, target_folder_path)