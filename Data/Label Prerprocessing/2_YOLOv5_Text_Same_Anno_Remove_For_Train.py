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
parser = argparse.ArgumentParser(description='YOLOv5_Text_Same_Anno_Remove_For_Train')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/Crowdman/labels', type=str, help='변경할 라벨들이 모여있는 폴더 지정')
parser.add_argument('--before-label', default="all", type=str, help='변경 이전 라벨 지정')
parser.add_argument('--after-label', default="2", type=str, help='변경 이후 라벨 지정')

args = parser.parse_args()

# ==============================================================
# 1. Label 파일명 추출 + Label 수정 (base_path -> train_path -> 각 label 변경)
# ==============================================================
def remove_same_anno(labels_path):
    # 1) Label 파일명 추출
    label = None
    for label_path in glob.glob(os.path.join(labels_path, '*.txt')):
        with open(label_path, 'r') as f:
            # 2) label 한줄씩 불러오기
            lines = f.readlines()

            # (1) 중복 anno 확인
            if len(set(lines)) != len(lines):
                print(f"lines : {len(lines)}")
                print(f"set(lines) : {len(set(lines))}")

            # (2) 중복 label+bbox 제거
            lines = list(set(lines))

        with open(label_path, 'w') as f:
            # 3) anno 저장
            for line in lines:
                # (1) label Split
                label, bbox = line.split(' ', maxsplit=1)
                # print(f'labels_path : {label_path} | label : {label}')
                # (2) anno 저장
                f.write(f'{label} {bbox}')

# ==============================================================
# 2. Main문
# ==============================================================
for idx, f_path in enumerate(['train/', 'val/', 'test/']):
    print(f'f_path : {f_path}')
    remove_same_anno(f'{args.base_path}/{f_path}', before_label = args.before_label, after_label = args.after_label)
