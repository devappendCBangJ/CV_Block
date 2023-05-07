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
parser = argparse.ArgumentParser(description='2_2_YOLOv5_Multi_Folder_Hierarchy_For_Train')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/Golfball_Near', type=str, help='옮길 데이터셋 폴더들이 존재하는 부모 폴더 경로 지정')
parser.add_argument('--move-path', default='_Move', type=str, help='데이터셋을 옮길 폴더 경로 지정')
parser.add_argument('--class-pathes', default=['golf ball.v6i.yolov5pytorch', 'GolfBallDetector.v10i.yolov5pytorch', 'golf ballGol - v2.v1i.yolov5pytorch', 'Golfwithme.v11-fixing.yolov5pytorch', 'Heimdallr.v1i.yolov5pytorch', 'PIDR.v6i.yolov5pytorch'], type=str, nargs='*', help='옮길 데이터셋 폴더명들 지정')
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
for class_path in args.class_pathes:
    for train_path in args.target_child_pathes:
        for image_path in args.target_parent_pathes:
            source_filenames = get_filenames(f'{args.base_path}/{class_path}/{train_path}/{image_path}', image_path)
            target_folder_path = f'{base_all_path}/{image_path}/{train_path}'
            for source_filename in source_filenames:
                source_file_path = f'{args.base_path}/{class_path}/{train_path}/{image_path}/{source_filename}'
                # print(source_file_path)
                # print(target_folder_path)
                if os.path.exists(f'{target_folder_path}/{source_filename}'):
                    print(f'중복 파일 : {target_folder_path}/{source_filename}')
                else:
                    shutil.move(source_file_path, target_folder_path)

# --------------------------------------------------------------
# 3) Source에서 파일명이 valid인 경우 val로 계층 이동하는 경우 따로 처리
# --------------------------------------------------------------
source_train_path = 'valid'
target_train_path = 'val'
for class_path in args.class_pathes:
    for image_path in args.target_parent_pathes:
        source_filenames = get_filenames(f'{args.base_path}/{class_path}/{source_train_path}/{image_path}', image_path)
        target_folder_path = f'{base_all_path}/{image_path}/{target_train_path}'
        for source_filename in source_filenames:
            source_file_path = f'{args.base_path}/{class_path}/{source_train_path}/{image_path}/{source_filename}'
            # print(source_file_path)
            # print(target_folder_path)
            if os.path.exists(f'{target_folder_path}/{source_filename}'):
                print(f'중복 파일 : {target_folder_path}/{source_filename}')
            else:
                shutil.move(source_file_path, target_folder_path)
