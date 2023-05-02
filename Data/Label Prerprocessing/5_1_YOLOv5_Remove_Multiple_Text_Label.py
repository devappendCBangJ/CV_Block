# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='4_3_YOLOv5_Remove_Multiple_Text_Label')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel_Remove_BboxInBbox_Remove_ErrorBboxRatio_Test', type=str, help='변경할 라벨들이 모여있는 grandmother 폴더 지정')
parser.add_argument('--source-parent-folders', default=['labels'], type=str, nargs='*', help='변경할 라벨들이 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='변경할 라벨들이 모여있는 child 폴더 지정')

parser.add_argument('--error-txt-filename', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel_Remove_BboxInBbox_Remove_ErrorBboxRatio/labels/error_bbox_ratio_handmade.txt', type=str, help='에러 파일명들을 모아둔 텍스트 파일 (단, 파일 확장자명을 제외한 파일명만 써져 있어야함)')

args = parser.parse_args()

# ==============================================================
# 0. 함수 정의
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# ==============================================================
# 1. 중복 Line 제거
# ==============================================================
def remove_label():
    # --------------------------------------------------------------
    # 1) Error Text 파일명 추출
    # --------------------------------------------------------------
    with open(args.error_txt_filename, 'r') as f:
        lines = f.readlines()

    # --------------------------------------------------------------
    # 2) Error Text 파일명의 label 제거
    # --------------------------------------------------------------
    lines = list(map(lambda s: s.strip(), lines))
    for label_folder in args.source_parent_folders:
        for train_folder in args.source_child_folders:
            for label_filename in lines:
                label_path = f'{args.base_path}/{label_folder}/{train_folder}/{label_filename}'
                if os.path.isfile(label_path):
                    print(f'error_label_filename : {label_filename}')
                    with open(label_path, 'w') as f:
                        pass

# ==============================================================
# 2. Main문
# ==============================================================
remove_label()