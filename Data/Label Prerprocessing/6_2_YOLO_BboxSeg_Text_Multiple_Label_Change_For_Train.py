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
parser = argparse.ArgumentParser(description='6_2_YOLO_BboxSeg_Text_Multiple_Label_Change_For_Train')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/4_1_LabelBang_AutoLabeling', type=str, help='변경할 다양한 종류의 라벨 폴더가 모여있는 부모 폴더 지정')
parser.add_argument('--bbox-seg-path', default=['bbox', 'seg'], type=str, help='각각의 종류의 라벨 폴더 내에 있는 Bbox, Seg 폴더명 지정')
parser.add_argument('--before-class', default="all", type=str, help='변경 이전 라벨 지정')
parser.add_argument('--after-class', default="0", type=str, help='변경 이후 라벨 지정')

args = parser.parse_args()

unique_cls = []

# ==============================================================
# 1. Label 파일명 추출 + Label 수정 (base_path -> train_path -> 각 label 변경)
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

def revise_label(labels_path, before_cls, after_cls):
    # 1) Label 파일명 추출
    for label_path in glob.glob(os.path.join(labels_path, '*.txt')):
        with open(label_path, 'r') as f:
            # 2) label 한줄씩 불러오기
            lines = f.readlines()

            # 3) class, label 값 확인
            for line in lines:
                cls, label = line.split(' ', maxsplit=1)
                # (1) Unique label 저장
                if cls not in unique_cls:
                    unique_cls.append(cls)
                # (2) before_label인 경우 확인
                if before_cls != "all" and cls == before_cls:
                    print(f'labels_path : {label_path} | class : {cls}')
                """
                # (3) 모든 경우 확인
                print(f'labels_path : {label_path} | class : {cls}')
                """

        with open(label_path, 'w') as f:
            # 4) label 변환
            for line in lines:
                # (1) label Split
                cls, label = line.split(' ', maxsplit=1)
                # print(f'labels_path : {label_path} | label : {label}')
                # (2) label 변환
                # 1] 전부 변환하는 경우
                if before_cls == "all":
                    f.write(f'{after_cls} {label}')
                # 2] 일부만 변환하는 경우
                else:
                    if cls == before_cls:
                        f.write(f'{after_cls} {label}')
                    else:
                        f.write(f'{cls} {label}')

    print(f'before unique_label : {unique_cls}')

# ==============================================================
# 2. Main문
# ==============================================================
base_folder_list = get_filenames(args.base_path)
for bf_path in base_folder_list:
    for bs_path in args.bbox_seg_path:
        print(f'f_path : {bf_path}/{bs_path}')
        revise_label(f'{args.base_path}/{bf_path}/{bs_path}', before_cls = args.before_class, after_cls = args.after_class)