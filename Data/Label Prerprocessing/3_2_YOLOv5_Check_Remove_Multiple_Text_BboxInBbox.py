# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse

from itertools import combinations

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='3_2_YOLOv5_Check_Remove_Multiple_Text_BboxInBbox')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Test', type=str, help='label의 오류를 탐지할 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-folders', default=['labels'], type=str, nargs='*', help='label의 오류를 탐지할 데이터셋이 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='label의 오류를 탐지할 데이터셋이 모여있는 child 폴더 지정')

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
def label_bboxinbbox_error_check_revise():
    # --------------------------------------------------------------
    # 1) Label 파일 불러오기 + 메모장 저장을 위한 변수 초기화
    # --------------------------------------------------------------
    for label_folder in args.source_parent_folders:
        for train_folder in args.source_child_folders:
            errors_info = {'| filename | idx |': ''}

            label_filenames = get_filenames(f'{args.base_path}/{label_folder}/{train_folder}')
            # --------------------------------------------------------------
            # 2) Line 한줄씩 Bbox 내에 Bbox 있는 라벨 확인
            # --------------------------------------------------------------
            for label_filename in label_filenames:
                # (1) 경로 추출
                label_path = f'{args.base_path}/{label_folder}/{train_folder}/{label_filename}'

                # (2) Bbox 내에 Bbox 있는 라벨 idx 추출
                with open(label_path, 'r') as f:
                    lines_info = []
                    delete_idx_list = []
                    # 1] Label 한줄씩 불러오기
                    lines = f.readlines()

                    # 2] Bbox 내에 있는 Bbox 탐지를 위한 정보 추출
                    for idx, line in enumerate(lines):
                        # [1] Label Split
                        _, x, y, w, h = line.split()

                        # [2] Label 자료형 변환
                        x, y, w, h = float(x), float(y), float(w), float(h)

                        # [3] Bbox 좌표 + 넓이 추출
                        x1, y1, x2, y2 = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)
                        area = w * h

                        # [4] Bbox 내에 있는 Bbox 탐지를 위한 정보 추출
                        lines_info.append({"idx":idx, "x1":x1, "y1":y1, "x2":x2, "y2":y2, "area":area})
                    # 3] Bbox 내에 Bbox 있는 라벨 idx 추출
                    # print(f"lines : {lines}")
                    for c1, c2 in combinations(lines_info, 2):
                        if c1["area"] >= c2["area"] and c1["x1"] < c2["x1"] and c1["y1"] < c2["y1"] and c1["x2"] > c2["x2"] and c1["y2"] > c2["y2"]:
                            delete_idx_list.append(c2["idx"])
                        elif c1["area"] < c2["area"] and c1["x1"] > c2["x1"] and c1["y1"] > c2["y1"] and c1["x2"] < c2["x2"] and c1["y2"] < c2["y2"]:
                            delete_idx_list.append(c1["idx"])
                    # print(f"lines : {lines}")
                # (3) Bbox 내에 Bbox 있는 라벨 idx 제거 + 해당 정보 저장
                if len(delete_idx_list) > 0:
                    with open(label_path, 'w') as f:
                        for idx, line in enumerate(lines):
                            if idx not in delete_idx_list:
                                f.write(line)
                    print(f"error_label_path : {label_path}")

                    errors_info[label_filename] = set()
                    errors_info[label_filename].update(delete_idx_list)
            # --------------------------------------------------------------
            # 3) 에러 정보 쓰기 (error_label_path + error_label_idx)
            # --------------------------------------------------------------
            error_bboxinbbox_txt_save_path = f'{args.base_path}/{label_folder}/error_bboxinbbox_{train_folder}.txt'
            with open(error_bboxinbbox_txt_save_path, 'w') as error_txt:
                error_txt.write(f'--------------------------------------------------------------\n')
                # (1) error_file_len
                error_txt.write(f'error_file_len : {len(errors_info)-1}\n')
                error_txt.write(f'--------------------------------------------------------------\n')
                # (2) error_label_path + error_line_num
                for error_label_path, error_line_idx in errors_info.items():
                    error_txt.write(f'{error_label_path} | {error_line_idx}\n')

label_bboxinbbox_error_check_revise()