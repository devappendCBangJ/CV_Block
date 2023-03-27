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

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/Merge_All', type=str, help='Plot할 데이터셋이 모여있는 폴더 경로 지정')
parser.add_argument('--source-parent-pathes', default=['images'], type=str, nargs='*', help='image_path')
parser.add_argument('--source-child-pathes', default=['train', 'val', 'test'], type=str, nargs='*', help='train_path')
parser.add_argument('--image-folder', default='images', type=str, help='image_folder')
parser.add_argument('--label-folder', default='labels', type=str, help='label_folder')
parser.add_argument('--before-file-extension', default='.jpg', type=str, help='before_file_extension')
parser.add_argument('--after-file-extension', default='.txt', type=str, help='after_file_extension')

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
                        x1 = (x - w / 2) * W
                        y1 = (y - h / 2) * H
                        x2 = (x + w / 2) * W
                        y2 = (y + h / 2) * H

                        # 4] Bounding Box 그리기
                        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

                        # 5] 텍스트 그리기
                        image = cv2.putText(image, label, org = (int(x1), int(y1)), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color = (255, 255, 255))

                # --------------------------------------------------------------
                # 3) Image 시각화
                # --------------------------------------------------------------
                cv2.imshow(f'{image_filename}', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

show_bbox(args.base_path)
