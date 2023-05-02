# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse

import cv2

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='4_1_YOLOv5_Check_Multiple_Text_ErrorBboxRatio')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel_Remove_BboxInBbox_Test', type=str, help='label의 오류를 탐지할 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-folders', default=['labels'], type=str, nargs='*', help='label의 오류를 탐지할 데이터셋이 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='label의 오류를 탐지할 데이터셋이 모여있는 child 폴더 지정')

parser.add_argument('--image-folder', default='images', type=str, help='base_path/source_parent_folders/source_child_folders에서 이미지가 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--label-folder', default='labels', type=str, help='base_path/source_parent_folders/source_child_folders에서 라벨이 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--before-file-extension', default='.txt', type=str, help='base_path/label_folder/source_child_folders 안에 들어있는 텍스트 파일 확장자')
parser.add_argument('--after-file-extension', default='.jpg', type=str, help='base_path/after_folder/source_child_folders 에서 오픈할 이미지 파일 확장자')

parser.add_argument('--error-bbox-ratio', default=2.5, type=int, help='제거할 bbox 비율 출력')

args = parser.parse_args()

# ==============================================================
# 0. 함수 정의
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# ==============================================================
# 1. 비정상 Label 추출
# ==============================================================
def label_error_bbox_ratio_check():
    # --------------------------------------------------------------
    # 1) Label 파일 불러오기 + 메모장 저장을 위한 변수 초기화
    # --------------------------------------------------------------
    for label_folder in args.source_parent_folders:
        for train_folder in args.source_child_folders:
            args.subplot_count = 0
            errors_info = {'| filename | idx |': ''}

            label_filenames = get_filenames(f'{args.base_path}/{label_folder}/{train_folder}')
            # --------------------------------------------------------------
            # 2) Line 한줄씩 Bbox 내에 Bbox 있는 라벨 확인
            # --------------------------------------------------------------
            for label_filename in label_filenames:
                # --------------------------------------------------------------
                # (1) 경로 추출 + Image 불러오기
                # --------------------------------------------------------------
                image_filename = label_filename.replace(args.before_file_extension, args.after_file_extension)
                image_path = f'{args.base_path}/{args.image_folder}/{train_folder}/{image_filename}'
                label_path = f'{args.base_path}/{args.label_folder}/{train_folder}/{label_filename}'

                image = cv2.imread(image_path)
                H, W, C = image.shape

                # --------------------------------------------------------------
                # (2) ErrorBboxRatio 라벨 idx 추출
                # --------------------------------------------------------------
                with open(label_path, 'r') as f:
                    # 1] Label 한줄씩 불러오기
                    lines = f.readlines()

                    # 2] Bbox 내에 있는 Bbox 탐지를 위한 정보 추출
                    for idx, line in enumerate(lines):
                        # [1] Label Split
                        _, _, _, w, h = line.split()

                        # [2] Label 자료형 변환
                        w, h = float(w), float(h)

                        # [3] Bbox 좌표 + 넓이 추출
                        real_w, real_h = W * w, H * h

                        # [4] ErrorBboxRatio 라벨 idx 추출
                        if max(real_w, real_h) >= min(real_w, real_h) * args.error_bbox_ratio:
                            if label_filename not in errors_info:
                                errors_info[label_filename] = set()
                            errors_info[label_filename].add(idx)

            # --------------------------------------------------------------
            # 3) 에러 정보 쓰기 (error_label_path + error_label_idx)
            # --------------------------------------------------------------
            error_bboxratio_txt_save_path = f'{args.base_path}/{args.label_folder}/error_bboxratio_{train_folder}.txt'
            with open(error_bboxratio_txt_save_path, 'w') as error_txt:
                error_txt.write(f'--------------------------------------------------------------\n')
                # (1) error_file_len
                error_txt.write(f'error_file_len : {len(errors_info)-1}\n')
                error_txt.write(f'--------------------------------------------------------------\n')
                # (2) error_label_path + error_line_num
                for error_label_path, error_line_idx in errors_info.items():
                    error_txt.write(f'{error_label_path} | {error_line_idx}\n')

label_error_bbox_ratio_check()