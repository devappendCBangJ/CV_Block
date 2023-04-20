# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
from PIL import Image, ImageDraw
import os
import argparse

import cv2

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='Object_Detection_PIL_Plot_BoundingBox')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/GSC/Human', type=str, help='Plot할 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-pathes', default=['images'], type=str, nargs='*', help='Plot할 데이터셋의 이미지가 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-pathes', default=['train', 'val', 'test'], type=str, nargs='*', help='Plot할 데이터셋의 이미지가 모여있는 child 폴더 지정')
parser.add_argument('--image-folder', default='images', type=str, help='base_path/source_parent_pathes/source_child_pathes에서 이미지가 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--label-folder', default='labels', type=str, help='base_path/source_parent_pathes/source_child_pathes에서 라벨이 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--before-file-extension', default='.jpg', type=str, help='base_path/image_folder/source_child_pathes 안에 들어있는 이미지 파일 확장자')
parser.add_argument('--after-file-extension', default='.txt', type=str, help='base_path/label_folder/source_child_pathes 에서 오픈할 텍스트 파일 확장자')

args = parser.parse_args()

# ==============================================================
# 1. Bounding Box 그리기
# ==============================================================
# --------------------------------------------------------------
# 1) 각 폴더 내 파일명 추출
# --------------------------------------------------------------
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

def show_bbox(base_path):
    # --------------------------------------------------------------
    # 1) Image Path + Label Path 정의
    # --------------------------------------------------------------
    for image_path in args.source_parent_pathes:
        for train_path in args.source_child_pathes:
            image_filenames = get_filenames(f'{args.base_path}/{image_path}/{train_path}')
            for image_filename in image_filenames:
                label_filename = image_filename.replace(args.before_file_extension, args.after_file_extension)

                image_path = f'{args.base_path}/{args.image_folder}/{train_path}/{image_filename}'
                label_path = f'{args.base_path}/{args.label_folder}/{train_path}/{label_filename}'

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
                        # 1] Label Split
                        label, x, y, w, h = line.split(' ')

                        # 2] Label 자료형 변환
                        x = float(x)
                        y = float(y)
                        w = float(w)
                        h = float(h)

                        # 3] Bounding Box 좌표 계산
                        H, W, C = image.shape
                        x1 = (x) * W
                        y1 = (y) * H
                        x2 = (x + w) * W
                        y2 = (y + h) * H

                        # 4] Bounding Box 그리기
                        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

                # --------------------------------------------------------------
                # 3) Image 시각화
                # --------------------------------------------------------------
                cv2.imshow(f'{image_filename}', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

show_bbox(args.base_path)
