# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse
import random
from copy import deepcopy

from itertools import combinations

import cv2
import matplotlib.pyplot as plt

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='3_3_YOLOv5_Check_Remove_Multiple_Text_BboxInBbox_Save_Plot_Bundle')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Test', type=str, help='label의 오류를 탐지할 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-folders', default=['labels'], type=str, nargs='*', help='label의 오류를 탐지할 데이터셋이 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='label의 오류를 탐지할 데이터셋이 모여있는 child 폴더 지정')

parser.add_argument('--image-folder', default='images', type=str, help='base_path/source_parent_folders/source_child_folders에서 이미지가 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--label-folder', default='labels', type=str, help='base_path/source_parent_folders/source_child_folders에서 라벨이 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--before-file-extension', default='.txt', type=str, help='base_path/label_folder/source_child_folders 안에 들어있는 텍스트 파일 확장자')
parser.add_argument('--after-file-extension', default='.jpg', type=str, help='base_path/after_folder/source_child_folders 에서 오픈할 이미지 파일 확장자')

parser.add_argument('--subplot-rows', default=4, type=int, help='Plot의 Row에 들어갈 이미지 개수')
parser.add_argument('--subplot-columns', default=4, type=int, help='Plot의 Column에 들어갈 이미지 개수')
parser.add_argument('--plt-after-save-folder', default='plt_after_images', type=str, help='전처리 전후 Plot 이미지를 저장할 mother 폴더 지정')
parser.add_argument('--plt-save-count', default=0, type=int, help='현재까지 저장한 이미지 개수')
parser.add_argument('--subplot-count', default=0, type=int, help='SubPlot을 위한 이미지 개수')

args = parser.parse_args()

unique_label = {}
plt.figure(figsize=(12, 12))

# ==============================================================
# 0. 함수 정의
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# ==============================================================
# 1. Plot한 이미지를 저장할 mother 폴더 생성
# ==============================================================
plt_after_save_folder_path = f'{args.base_path}/{args.plt_after_save_folder}'
if not os.path.exists(plt_after_save_folder_path):
    os.makedirs(plt_after_save_folder_path)

# ==============================================================
# 2. 비정상 Label 추출
# ==============================================================
def label_bboxinbbox_error_check_revise():
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
                # (1) 경로 추출
                # --------------------------------------------------------------
                label_path = f'{args.base_path}/{args.label_folder}/{train_folder}/{label_filename}'

                # --------------------------------------------------------------
                # (2) Bbox 내에 Bbox 있는 라벨 idx 추출
                # --------------------------------------------------------------
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

                # --------------------------------------------------------------
                # (3) Bbox 내에 Bbox 있는 라벨 idx 제거 & 해당 정보 저장 + 수정 이전 & 이후 Image Bundle Bbox 그리기
                # --------------------------------------------------------------
                if len(delete_idx_list) > 0:
                    # 0] 변수 선언
                    after_lines = []

                    # 1] Bbox 내에 Bbox 있는 라벨 idx 제거 & 해당 정보 저장
                    with open(label_path, 'w') as f:
                        for idx, line in enumerate(lines):
                            if idx not in delete_idx_list:
                                after_lines.append(line)
                                f.write(line)
                    print(f"error_label_path : {label_path}")

                    errors_info[label_filename] = set()
                    errors_info[label_filename].update(delete_idx_list)

                    # 2] Image 불러오기
                    image_filename = label_filename.replace(args.before_file_extension, args.after_file_extension)
                    image_path = f'{args.base_path}/{args.image_folder}/{train_folder}/{image_filename}'

                    image = cv2.imread(image_path)

                    # 3] Label 한줄씩 불러오기 + 수정 이전 & 이후 Image Bundle Bbox 그리기
                    H, W, C = image.shape
                    for iter_lines in [lines, after_lines]:
                        image_copy = image.copy()
                        for line in iter_lines:
                            try:
                                # [1] Label Split
                                label, x, y, w, h = line.split(' ')

                                # [2] unique label : 서로 다른 색상 사용
                                if label not in unique_label:
                                    temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50),
                                                  random.randrange(0, 255, 50))
                                    while temp_color in unique_label.values():
                                        temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50),
                                                      random.randrange(0, 255, 50))
                                    unique_label[label] = temp_color

                                # [3] Label 자료형 변환
                                x, y, w, h = float(x), float(y), float(w), float(h)

                                # [4] Bounding Box 좌표 계산
                                x1, y1, x2, y2 = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H

                                # [5] Bounding Box 그리기
                                image_copy = cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)),
                                                      color=unique_label[label], thickness=5)

                                # [6] 텍스트 그리기
                                image_copy = cv2.putText(image_copy, label, org=(int(x1), int(y1)),
                                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                    fontScale=2, color=unique_label[label])
                            except:
                                print(label_path)

                        # 4] 수정 이전 & 이후 Image Bundle Bbox 저장
                        # [1] Plt Subplot Split
                        plot_idx = args.subplot_count % (args.subplot_rows * args.subplot_columns)
                        plt.subplot(args.subplot_rows, args.subplot_columns, plot_idx + 1)

                        # [2] Plt Title
                        plt.title(image_filename, fontsize=5)

                        # [3] Plt Label
                        # plt.xlabel('x-axis')
                        # plt.ylabel('y-axis')

                        plt.xticks([])
                        plt.yticks([])

                        # [4] Subplot에 Image 저장
                        plt.imshow(image_copy[:, :, ::-1])

                        # [5] Plt Figure Axis
                        plt.axis("off")

                        # [6] Image 시각화
                        if plot_idx == (args.subplot_rows * args.subplot_columns - 1) and args.subplot_count != 0:
                            # plt.show()
                            args.plt_save_count += 1
                            plt_after_save_path = f'{plt_after_save_folder_path}/{args.plt_save_count}.png'
                            plt.savefig(plt_after_save_path, facecolor='#eeeeee', dpi=100, edgecolor='black')

                        args.subplot_count += 1

            # --------------------------------------------------------------
            # 2) 잔여 Image 시각화
            # --------------------------------------------------------------
            if plot_idx != (args.subplot_rows * args.subplot_columns - 1) and args.subplot_count != 0:
                args.plt_save_count += 1
                plt_after_save_path = f'{plt_after_save_folder_path}/{args.plt_save_count}.png'
                plt.savefig(plt_after_save_path, facecolor='#eeeeee', dpi=100, edgecolor='black')

            # --------------------------------------------------------------
            # 3) 에러 정보 쓰기 (error_label_path + error_label_idx)
            # --------------------------------------------------------------
            error_line_len_txt_save_path = f'{args.base_path}/{args.label_folder}/error_line_len_{train_folder}.txt'
            with open(error_line_len_txt_save_path, 'w') as error_txt:
                error_txt.write(f'--------------------------------------------------------------\n')
                # (1) error_file_len
                error_txt.write(f'error_file_len : {len(errors_info)-1}\n')
                error_txt.write(f'--------------------------------------------------------------\n')
                # (2) error_label_path + error_line_num
                for error_label_path, error_line_idx in errors_info.items():
                    error_txt.write(f'{error_label_path} | {error_line_idx}\n')

label_bboxinbbox_error_check_revise()