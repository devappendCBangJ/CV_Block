# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import glob
import argparse

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='YOLOv5_Text_Multiple_Label_Check_Same_Anno_Check')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near/labels', type=str, help='변경할 라벨들이 모여있는 폴더 지정')
parser.add_argument('--before-label', default="all", type=str, help='변경 이전 라벨 지정')
parser.add_argument('--after-label', default="0", type=str, help='변경 이후 라벨 지정')

args = parser.parse_args()

unique_label = []

# ==============================================================
# 1. Label 파일명 추출 + Label 수정 (base_path -> train_path -> 각 label 변경)
# ==============================================================
def revise_label(labels_path, before_label, after_label):
    # 1) Label 파일명 추출
    label = None
    for label_path in glob.glob(os.path.join(labels_path, '*.txt')):
        with open(label_path, 'r') as f:
            # 2) label 한줄씩 불러오기
            lines = f.readlines()

            # 3) label, bbox 값 확인
            for line in lines:
                label, bbox = line.split(' ', maxsplit=1)
                # (1) Unique label 저장
                if label not in unique_label:
                    unique_label.append(label)
                # (2) before_label인 경우 확인
                if before_label != "all" and label == before_label:
                    print(f'labels_path : {label_path} | label : {label}')
                """
                # (3) 모든 경우 확인
                print(f'labels_path : {label_path} | label : {label}')
                """

        with open(label_path, 'w') as f:
            # 4) label 변환
            for line in lines:
                # (1) label Split
                label, bbox = line.split(' ', maxsplit=1)
                # print(f'labels_path : {label_path} | label : {label}')
                # (2) label 변환
                # 1] 전부 변환하는 경우
                if before_label == "all":
                    f.write(f'{after_label} {bbox}')
                # 2] 일부만 변환하는 경우
                else:
                    if label == before_label:
                        f.write(f'{after_label} {bbox}')
                    else:
                        f.write(f'{label} {bbox}')

    print(f'unique_label : {unique_label}')

# ==============================================================
# 2. Main문
# ==============================================================
for idx, f_path in enumerate(['train/', 'val/', 'test/']):
    print(f'f_path : {f_path}')
    revise_label(f'{args.base_path}/{f_path}', before_label = args.before_label, after_label = args.after_label)
