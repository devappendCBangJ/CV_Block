# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
from PIL import Image, ImageDraw

# ==============================================================
# 1. Bounding Box 그리기
# ==============================================================
def show_bbox(image_path):
    # --------------------------------------------------------------
    # 1) Image Path + Label Path 정의
    # --------------------------------------------------------------
    label_path = image_path.replace('/images/', '/darknet/')
    label_path = label_path.replace('.jpg', '.txt')

    # --------------------------------------------------------------
    # 2) Image 불러오기 + Bounding Box 그리기 준비
    # --------------------------------------------------------------
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

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
            W, H = image.size
            x1 = (x - w / 2) * W
            y1 = (y - h / 2) * H
            x2 = (x + w / 2) * W
            y2 = (y + h / 2) * H

            # 4] Bounding Box 그리기
            draw.rectangle((x1, y1, x2, y2),
                           outline=(255, 0, 0),  # Red in RGB
                           width=5)  # Line width

    # --------------------------------------------------------------
    # 3) Image 시각화
    # --------------------------------------------------------------
    image.show()

show_bbox('./Dataset/Cat_Dog/cat/images/0a0df46ca3f886c9.jpg')
