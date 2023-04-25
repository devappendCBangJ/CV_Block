# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse
import random

import cv2

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='0_YOLOv5_OpenCV_Save_CenterCoordinates_MultipleBoundingBox_Print_Error')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_Test', type=str, help='Plot할 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-folders', default=['images'], type=str, nargs='*', help='Plot할 데이터셋의 이미지가 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='Plot할 데이터셋의 이미지가 모여있는 child 폴더 지정')
parser.add_argument('--image-folder', default='images', type=str, help='base_path/source_parent_folders/source_child_folders에서 이미지가 모여있는 source_parent_folders 폴더명')
parser.add_argument('--label-folder', default='labels', type=str, help='base_path/source_parent_folders/source_child_folders에서 라벨이 모여있는 source_parent_folders 폴더명')
parser.add_argument('--before-file-extension', default='.jpg', type=str, help='base_path/image_folder/source_child_folders 안에 들어있는 이미지 파일 확장자')
parser.add_argument('--after-file-extension', default='.txt', type=str, help='base_path/label_folder/source_child_folders 에서 오픈할 텍스트 파일 확장자')

args = parser.parse_args()

unique_label = {}

# ==============================================================
# 1. Bounding Box 그리기
# ==============================================================
# --------------------------------------------------------------
# 1) 각 폴더 내 파일명 추출
# --------------------------------------------------------------
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

def draw_bbox():
    base_DrawBbox_path = args.base_path + "_DrawBbox"
    if not os.path.exists(base_DrawBbox_path):
        for image_folder in args.source_parent_folders:
            for train_folder in args.source_child_folders:
                os.makedirs(f'{base_DrawBbox_path}/{image_folder}/{train_folder}')
                os.makedirs(f'{base_DrawBbox_path}/labels/{train_folder}')

    # --------------------------------------------------------------
    # 0) 변수 선언
    # --------------------------------------------------------------
    errors_info = ['| filename | cls | x | y | img_w | img_h |']

    # --------------------------------------------------------------
    # 1) Image Path + Label Path 정의
    # --------------------------------------------------------------
    for image_folder in args.source_parent_folders:
        for train_folder in args.source_child_folders:
            image_filenames = get_filenames(f'{args.base_path}/{image_folder}/{train_folder}')
            image_filenames.sort()
            for image_filename in image_filenames:
                label_filename = image_filename.replace(args.before_file_extension, args.after_file_extension)

                image_path = f'{args.base_path}/{args.image_folder}/{train_folder}/{image_filename}'
                label_path = f'{args.base_path}/{args.label_folder}/{train_folder}/{label_filename}'
                image_DrawBbox_path = f'{base_DrawBbox_path}/{args.image_folder}/{train_folder}/{image_filename}'

                # --------------------------------------------------------------
                # 2) Image 불러오기 + Bounding Box 그리기 준비
                # --------------------------------------------------------------
                image = cv2.imread(image_path)

                # --------------------------------------------------------------
                # 3) 시도 : Label 불러오기 + Bounding Box 그리기
                # --------------------------------------------------------------
                try:
                    with open(label_path, 'r') as f:
                        # --------------------------------------------------------------
                        # (1) Label 한줄씩 불러오기 -> Bbox 그리기
                        # --------------------------------------------------------------
                        for line in f.readlines():
                            # 1] Label Split
                            print(label_path)
                            label, x, y, w, h = line.split(' ')

                            # 2] unique label : 서로 다른 색상 사용
                            if label not in unique_label:
                                temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50), random.randrange(0, 255, 50))
                                while temp_color in unique_label.values():
                                    temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50), random.randrange(0, 255, 50))
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
                            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=unique_label[label], thickness=3)

                            # 5] 텍스트 그리기
                            image = cv2.putText(image, label, org = (int(x1), int(y1)), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 2, color = unique_label[label])

                    # --------------------------------------------------------------
                    # (2) Image 저장
                    # --------------------------------------------------------------
                    cv2.imwrite(image_DrawBbox_path, image)
                # --------------------------------------------------------------
                # 4) Label 불러오기 or bbox 불러오기 실패 시 : 에러 목록 저장
                # --------------------------------------------------------------
                except:
                    errors_info.append(label_path)

            # --------------------------------------------------------------
            # 5) 에러 목록 쓰기 + 초기화
            # --------------------------------------------------------------
            DrawBbox_error_txt_save_path = f'{base_DrawBbox_path}/{args.label_folder}/{train_folder}/error_{train_folder}.txt'
            with open(DrawBbox_error_txt_save_path, 'w') as error_txt:
                for error_info in errors_info:
                    error_txt.write('%s\n' % error_info)
            errors_info = ['| filename | cls | x | y | img_w | img_h |']

draw_bbox()