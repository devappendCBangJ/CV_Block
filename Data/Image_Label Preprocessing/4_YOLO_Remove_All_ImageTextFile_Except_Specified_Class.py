# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import argparse
from tqdm import tqdm

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='6_1_YOLOv5_Text_Multiple_Label_Change_For_Train')

parser.add_argument('--labels-base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/1_3_COCO_OnlyBall_SubHuman/labels', type=str, help='검토할 라벨들이 모여있는 부모 폴더 지정')
parser.add_argument('--images-base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/1_3_COCO_OnlyBall_SubHuman/images', type=str, help='검토할 이미지들이 모여있는 부모 폴더 지정')
parser.add_argument('--train-val-dirs', default=['train', 'val', 'test'], type=str, help='검토할 라벨들이 모여있는 자식 폴더 지정')
parser.add_argument('--main-cls', default="37", type=str, help='남길 클래스 지정(여기에 해당하는 클래스를 제외한 레이블만 가지고 있는 txt, 이미지 파일은 전부 제거)')
parser.add_argument('--before-file-extension', default='.txt', type=str, help='source_label_parent_path/child_path 에서 오픈할 텍스트 파일 확장자')
parser.add_argument('--after-file-extension', default='.jpg', type=str, help='source_image_parent_path/child_path 안에 들어있는 이미지 파일 확장자')

args = parser.parse_args()

# ==============================================================
# 1. 폴더 내 파일 리스트 추출
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# ==============================================================
# 2. Label 파일명 추출 + Label 수정 (base_path -> train_path -> 각 label 변경)
# ==============================================================
def remain_main_label_file(labels_base_path, images_base_path, train_val_dir, main_cls):
    # 0) 초기 세팅
    labels_parent_abs_path = f"{labels_base_path}/{train_val_dir}"
    images_parent_abs_path = f"{images_base_path}/{train_val_dir}"
    removed_txts_info = []
    # 1) Label 파일명 추출
    for label_filename in tqdm(get_filenames(labels_parent_abs_path)):
        label_abs_path = f"{labels_parent_abs_path}/{label_filename}"
        with open(label_abs_path, 'r') as f:
            # 2) label 한줄씩 불러오기
            lines = f.readlines()
            # 3) Unique Class 추출
            unique_cls = {line.split(' ', maxsplit=1)[0] for line in lines}
            # 4-1) Unique Class에 원하는 Main Class 있는 경우 : Continue
            if main_cls in unique_cls:
                continue
            # 4-2) Unique Class에 원하는 Main Class 없는 경우
            else:
                # (1) 해당 Label + Image 제거
                os.unlink(label_abs_path)
                image_abs_path = f"{images_parent_abs_path}/{label_filename.replace(args.before_file_extension, args.after_file_extension)}"
                os.unlink(image_abs_path)

                # (2) 제거한 파일 경로 모아두기
                removed_txts_info.append(label_abs_path)
    # 5) 제거한 파일 경로 저장
    removed_txt_abs_path = f"{args.labels_base_path}/removed_{train_val_dir}.txt"
    with open(removed_txt_abs_path, 'w') as removed_txt:
        for removed_txt_info in removed_txts_info:
            removed_txt.write('%s\n' % removed_txt_info)

# ==============================================================
# 0. Main문
# ==============================================================
for train_val_dir in args.train_val_dirs:
    print(f'train_val_dir : {train_val_dir}')
    remain_main_label_file(labels_base_path = args.labels_base_path, images_base_path = args.images_base_path, train_val_dir = train_val_dir, main_cls = args.main_cls)
