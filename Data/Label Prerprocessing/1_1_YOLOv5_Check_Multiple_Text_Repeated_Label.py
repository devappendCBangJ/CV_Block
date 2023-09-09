# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='1_1_YOLOv5_Check_Multiple_Text_Repeated_Label')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel _Test', type=str, help='변경할 라벨들이 모여있는 grandmother 폴더 지정')
parser.add_argument('--source-parent-folders', default=['labels'], type=str, nargs='*', help='변경할 라벨들이 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='변경할 라벨들이 모여있는 child 폴더 지정')

args = parser.parse_args()

unique_label = []

# ==============================================================
# 0. 함수 정의
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# ==============================================================
# 1. 중복 Line 추출
# ==============================================================
def revise_label():
    # --------------------------------------------------------------
    # 1) Label 파일명 추출
    # --------------------------------------------------------------
    for label_folder in args.source_parent_folders:
        for train_folder in args.source_child_folders:
            label_filenames = get_filenames(f'{args.base_path}/{label_folder}/{train_folder}')
            for label_filename in label_filenames:
                label_path = f'{args.base_path}/{label_folder}/{train_folder}/{label_filename}'
                with open(label_path, 'r') as f:
                    # --------------------------------------------------------------
                    # 2) Label 한줄씩 불러오기
                    # --------------------------------------------------------------
                    lines = f.readlines()

                    # --------------------------------------------------------------
                    # 3) 중복 Line 확인
                    # --------------------------------------------------------------
                    if len(set(lines)) != len(lines):
                        print(f"label_filename : {label_filename}")
                        print(f"lines : {lines}")
                        print(f"set(lines) : {set(lines)}")

# ==============================================================
# 2. Main문
# ==============================================================
revise_label()