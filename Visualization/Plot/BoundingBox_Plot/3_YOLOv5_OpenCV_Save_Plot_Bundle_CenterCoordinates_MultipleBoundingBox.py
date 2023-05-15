# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse
import random

import cv2
import matplotlib.pyplot as plt

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='3_YOLOv5_OpenCV_Save_Plot_Bundle_CenterCoordinates_MultipleBoundingBox')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel', type=str, help='Plot할 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-folders', default=['images'], type=str, nargs='*', help='Plot할 데이터셋의 이미지가 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='Plot할 데이터셋의 이미지가 모여있는 child 폴더 지정')
parser.add_argument('--image-folder', default='images', type=str, help='base_path/source_parent_folders/source_child_folders에서 이미지가 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--label-folder', default='labels', type=str, help='base_path/source_parent_folders/source_child_folders에서 라벨이 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--before-file-extension', default='.jpg', type=str, help='base_path/image_folder/source_child_folders 안에 들어있는 이미지 파일 확장자')
parser.add_argument('--after-file-extension', default='.txt', type=str, help='base_path/label_folder/source_child_folders 에서 오픈할 텍스트 파일 확장자')

parser.add_argument('--subplot-rows', default=4, type=int, help='Plot의 Row에 들어갈 이미지 개수')
parser.add_argument('--subplot-columns', default=4, type=int, help='Plot의 Column에 들어갈 이미지 개수')

parser.add_argument('--plt-save-folder', default='plt_images', type=str, help='Plot한 이미지를 저장할 mother 폴더 지정')
parser.add_argument('--plt-save-count', default=0, type=int, help='현재까지 저장한 이미지 개수')

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
plt_save_folder_path = f'{args.base_path}/{args.plt_save_folder}'
if not os.path.exists(plt_save_folder_path):
    os.makedirs(plt_save_folder_path)

# ==============================================================
# 2. Bounding Box 그리기 + 묶음 저장
# ==============================================================
def show_save_bboxes():
    # --------------------------------------------------------------
    # 1) Image Path + Label Path 정의
    # --------------------------------------------------------------
    for image_folder in args.source_parent_folders:
        for train_folder in args.source_child_folders:
            args.subplot_count = 0
            image_filenames = get_filenames(f'{args.base_path}/{image_folder}/{train_folder}')
            for i, image_filename in enumerate(image_filenames):
                label_filename = image_filename.replace(args.before_file_extension, args.after_file_extension)

                image_path = f'{args.base_path}/{args.image_folder}/{train_folder}/{image_filename}'
                label_path = f'{args.base_path}/{args.label_folder}/{train_folder}/{label_filename}'

                # --------------------------------------------------------------
                # 2) Image 불러오기 + Bounding Box 그리기 준비
                # --------------------------------------------------------------
                image = cv2.imread(image_path)

                # --------------------------------------------------------------
                # 3) Label 불러오기 + Bounding Box 그리기
                # --------------------------------------------------------------
                with open(label_path, 'r') as f:
                    # (1) Label 한줄씩 불러오기
                    for line in f.readlines():
                        try:
                            # 1] Label Split
                            label, x, y, w, h = line.split(' ')

                            # 2] unique label : 서로 다른 색상 사용
                            if label not in unique_label:
                                temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50),
                                              random.randrange(0, 255, 50))
                                while temp_color in unique_label.values():
                                    temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50),
                                                  random.randrange(0, 255, 50))
                                unique_label[label] = temp_color

                            # 2] Label 자료형 변환
                            x = float(x)
                            y = float(y)
                            w = float(w)
                            h = float(h)

                            # 3] Bounding Box 좌표 계산
                            H, W, C = image.shape
                            x1 = (x - w / 2) * W
                            y1 = (y - h / 2) * H
                            x2 = (x + w / 2) * W
                            y2 = (y + h / 2) * H

                            # 4] Bounding Box 그리기
                            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                                                  color=unique_label[label], thickness=5)

                            # 5] 텍스트 그리기
                            image = cv2.putText(image, label, org=(int(x1), int(y1)), fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                fontScale=2, color=unique_label[label])
                        except:
                            print(label_path)

                # --------------------------------------------------------------
                # 4) Image Bundle 저장
                # --------------------------------------------------------------
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
                plt.imshow(image[:, :, ::-1])

                # [5] Plt Figure Axis
                plt.axis("off")

                # (6) Image 시각화
                if plot_idx == (args.subplot_rows * args.subplot_columns - 1) and i != 0:
                    # plt.show()
                    args.plt_save_count += 1
                    plt_save_path = f'{plt_save_folder_path}/{args.plt_save_count}.png'
                    plt.savefig(plt_save_path, facecolor='#eeeeee', dpi = 100, edgecolor='black')

                args.subplot_count += 1

            # --------------------------------------------------------------
            # 5) 잔여 Image 시각화
            # --------------------------------------------------------------
            if plot_idx != (args.subplot_rows * args.subplot_columns - 1) and args.subplot_count != 0:
                args.plt_save_count += 1
                plt_save_path = f'{plt_save_folder_path}/{args.plt_save_count}.png'
                plt.savefig(plt_save_path, facecolor='#eeeeee', dpi=100, edgecolor='black')

show_save_bboxes()