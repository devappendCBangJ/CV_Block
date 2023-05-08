# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import shutil
import argparse

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='4_YOLOv5_File_Merge_For_Train')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone', type=str, help='옮길 데이터셋 폴더들이 존재하는 부모 폴더 경로 지정')
parser.add_argument('--merge-path', default='/Merge_All', type=str, help='Merge 데이터셋을 저장할 폴더 경로 지정')
parser.add_argument('--class-pathes', default=['CrowdHuman', 'Golfball_Near'], type=str, nargs='*', help='옮길 데이터셋 폴더명들 지정')
parser.add_argument('--source-parent-pathes', default=['images', 'labels'], type=str, nargs='*', help='source 폴더 기준 부모 폴더들 경로')
parser.add_argument('--source-child-pathes', default=['train', 'val', 'test'], type=str, nargs='*', help='source 폴더 기준 자식 폴더들 경로')

args = parser.parse_args()

# ==============================================================
# 1. 파일 이동을 위한 폴더 생성
# ==============================================================
base_all_path = args.base_path + args.merge_path
if not os.path.exists(base_all_path):
    for folder in args.source_parent_pathes:
        for split in args.source_child_pathes:
            os.makedirs(f'{base_all_path}/{folder}/{split}')

# ==============================================================
# 2. 폴더 내 파일 합치기
# ==============================================================
# --------------------------------------------------------------
# 1) 각 폴더 내 파일명 추출
# --------------------------------------------------------------
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# --------------------------------------------------------------
# 2) 폴더 내 파일 합치기 (각 class_path -> image_path -> train_path 순회하면서 base_all_path -> image_path -> train_path으로 합침)
# --------------------------------------------------------------
for class_path in args.class_pathes:
    for image_path in args.source_parent_pathes:
        for train_path in args.source_child_pathes:
            folder_abs_path = f'{args.base_path}/{class_path}/{image_path}/{train_path}'
            # (1) 폴더 존재하는 경우
            if os.path.exists(folder_abs_path):
                # 1] source 폴더 내 파일명 추출
                source_filenames = get_filenames(folder_abs_path)
                # 2] target 폴더로 파일 복사
                target_folder_path = f'{base_all_path}/{image_path}/{train_path}'
                for source_filename in source_filenames:
                    source_file_path = f'{args.base_path}/{class_path}/{image_path}/{train_path}/{source_filename}'
                    # 3] 복사한 파일 경로 시각화
                    print(f'source_file_path : {source_file_path}')
                    print(f'target_folder_path : {target_folder_path}')
                    shutil.copy(source_file_path, target_folder_path)
