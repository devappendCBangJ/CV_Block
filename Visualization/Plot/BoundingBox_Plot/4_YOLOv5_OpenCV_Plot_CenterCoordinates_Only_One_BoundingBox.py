# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import argparse
import random

import cv2

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='4_YOLOv5_PIL_Plot_CenterCoordinates_Only_One_BoundingBox')

parser.add_argument('--image-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel_Remove_BboxInBbox_Check_ErrorBboxRatio/images/train/0-9593-276_jpg.rf.cd034c928f33279d7d2153700458b553.jpg', type=str, help='Plot할 이미지 1개 경로')

args = parser.parse_args()

unique_label = {}

# ==============================================================
# 1. Bounding Box 그리기
# ==============================================================
def show_bbox(image_path):
    # --------------------------------------------------------------
    # 1) Image Path + Label Path 정의
    # --------------------------------------------------------------
    label_path = image_path.replace('/images/', '/labels/')
    label_path = label_path.replace('.jpg', '.txt')

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

            # 2] unique label : 서로 다른 색상 사용
            if label not in unique_label:
                temp_color = (random.randrange(0, 255, 50), random.randrange(0, 255, 50), random.randrange(0, 255, 50))
                while temp_color in unique_label.values():
                    temp_color = (
                    random.randrange(0, 255, 50), random.randrange(0, 255, 50), random.randrange(0, 255, 50))
                unique_label[label] = temp_color

            # 3] Label 자료형 변환
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            # 4] Bounding Box 좌표 계산
            H, W, C = image.shape
            x1 = (x - w / 2) * W
            y1 = (y - h / 2) * H
            x2 = (x + w / 2) * W
            y2 = (y + h / 2) * H

            # 5] Bounding Box 그리기
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=unique_label[label], thickness=3)

            # 6] 텍스트 그리기
            image = cv2.putText(image, label, org=(int(x1), int(y1)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=unique_label[label])

    # --------------------------------------------------------------
    # 3) Image 시각화
    # --------------------------------------------------------------
    cv2.imshow(args.image_path.split('/')[-1], image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show_bbox(args.image_path)
