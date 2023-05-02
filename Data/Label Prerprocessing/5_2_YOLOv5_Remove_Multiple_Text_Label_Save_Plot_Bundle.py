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
parser = argparse.ArgumentParser(description='4_3_YOLOv5_Remove_Multiple_Text_Label')

parser.add_argument('--error-txt-filename', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel_Remove_BboxInBbox_Remove_ErrorBboxRatio/labels/error_bbox_ratio_handmade.txt', type=str, help='에러 파일명들을 모아둔 텍스트 파일 (단, 파일 확장자명을 제외한 파일명만 써져 있어야함)')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel_Remove_BboxInBbox_Remove_ErrorBboxRatio', type=str, help='변경할 라벨들이 모여있는 grandmother 폴더 지정')
parser.add_argument('--source-parent-folders', default=['labels'], type=str, nargs='*', help='변경할 라벨들이 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='변경할 라벨들이 모여있는 child 폴더 지정')

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
# 1. 중복 Line 제거
# ==============================================================
def remove_label():
    # --------------------------------------------------------------
    # 1) Error Text 파일명 추출
    # --------------------------------------------------------------
    with open(args.error_txt_filename, 'r') as f:
        label_filenames = f.readlines()

    # --------------------------------------------------------------
    # 2) Error Text 파일명 존재하는지 확인
    # --------------------------------------------------------------
    label_filenames = list(map(lambda s: s.strip(), label_filenames))
    for label_folder in args.source_parent_folders:
        for train_folder in args.source_child_folders:
            args.subplot_count = 0
            for label_filename in label_filenames:
                label_path = f'{args.base_path}/{label_folder}/{train_folder}/{label_filename}'
                if os.path.isfile(label_path):
                    print(f'error_label_filename : {label_filename}')

                    # --------------------------------------------------------------
                    # 3) Image 불러오기 + Label 한줄씩 불러오기
                    # --------------------------------------------------------------
                    image_filename = label_filename.replace(args.before_file_extension, args.after_file_extension)
                    image_path = f'{args.base_path}/{args.image_folder}/{train_folder}/{image_filename}'

                    image = cv2.imread(image_path)
                    H, W, C = image.shape

                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    # --------------------------------------------------------------
                    # 4) Error Text 파일명의 Label 제거
                    # --------------------------------------------------------------
                    with open(label_path, 'w') as f:
                        pass

                    # --------------------------------------------------------------
                    # 5) Label 한줄씩 불러오기 + 수정 이전 & 이후 Image Bundle Bbox 그리기
                    # --------------------------------------------------------------
                    image_copy = image.copy()
                    for line in lines:
                        try:
                            # (1) Label Split
                            label, x, y, w, h = line.split(' ')

                            # (2) unique label : 서로 다른 색상 사용
                            if label not in unique_label:
                                temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50),
                                              random.randrange(0, 255, 50))
                                while temp_color in unique_label.values():
                                    temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50),
                                                  random.randrange(0, 255, 50))
                                unique_label[label] = temp_color

                            # (3) Label 자료형 변환
                            x, y, w, h = float(x), float(y), float(w), float(h)

                            # (4) Bounding Box 좌표 계산
                            x1, y1, x2, y2 = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H

                            # (5) Bounding Box 그리기
                            image_copy = cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)),
                                                       color=unique_label[label], thickness=5)

                            # (6) 텍스트 그리기
                            image_copy = cv2.putText(image_copy, label, org=(int(x1), int(y1)),
                                                     fontFace=cv2.FONT_HERSHEY_PLAIN,
                                                     fontScale=2, color=unique_label[label])
                        except:
                            print(label_path)

                    # --------------------------------------------------------------
                    # 6) 수정 이전 & 이후 Image Bundle Bbox 저장
                    # --------------------------------------------------------------
                    for img in [image_copy, image]:
                        # (1) Plt Subplot Split
                        plot_idx = args.subplot_count % (args.subplot_rows * args.subplot_columns)
                        plt.subplot(args.subplot_rows, args.subplot_columns, plot_idx + 1)

                        # (2) Plt Title
                        plt.title(image_filename, fontsize=5)

                        # (3) Plt Label
                        # plt.xlabel('x-axis')
                        # plt.ylabel('y-axis')

                        plt.xticks([])
                        plt.yticks([])

                        # (4) Subplot에 Image 저장
                        plt.imshow(img[:, :, ::-1])

                        # (5) Plt Figure Axis
                        plt.axis("off")

                        # (6) Image 시각화
                        if plot_idx == (args.subplot_rows * args.subplot_columns - 1) and args.subplot_count != 0:
                            # plt.show()
                            args.plt_save_count += 1
                            plt_after_save_path = f'{plt_after_save_folder_path}/{args.plt_save_count}.png'
                            plt.savefig(plt_after_save_path, facecolor='#eeeeee', dpi=100, edgecolor='black')

                        args.subplot_count += 1

# ==============================================================
# 2. Main문
# ==============================================================
remove_label()