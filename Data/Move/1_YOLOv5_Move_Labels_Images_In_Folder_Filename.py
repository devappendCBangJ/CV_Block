# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import shutil
import argparse

# ==============================================================
# 0. 변수 정의
# ==============================================================
# Image와 동일한 파일명을 가진 텍스트 파일을 찾은 후, 현재 Image 파일 경로와 대응되는 폴더 위치로 옮기는 코드
parser = argparse.ArgumentParser(description='1_Move_Labels_Images_In_Folder_Filename')

parser.add_argument('--extract-folder', default='/media/hi/SK Gold P31/Capstone/GolfBall/Val_Images', type=str, help='원하는 파일명들이 모여있는 폴더 지정')
parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_FixLabel_Remove_BboxInBbox_Remove_ErrorBboxRatio', type=str, help='라벨과 이미지 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--search-child-folders', default=['train', 'test'], type=str, nargs='*', help='탐색할 이미지가 있는 자식 폴더명')
parser.add_argument('--target-child-folder', default='val', type=str, help='target 자식 폴더명')

parser.add_argument('--image-file-extension', default='.jpg', type=str, help='base_path/label_folder/child_folders 안에 들어있는 이미지 파일 확장자')
parser.add_argument('--label-file-extension', default='.txt', type=str, help='base_path/after_folder/child_folders으로 옮길 텍스트 파일 확장자')

args = parser.parse_args()

# ==============================================================
# 1. 파일명 추출
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# ==============================================================
# 2. 이미지 파일 추출 -> 동일한 파일명의 Label을 이미지 폴더와 대응되는 위치로 이동
# ==============================================================
def move_labels_corresponding_images():
    # 1) 이미지 파일명 추출
    image_filenames = get_filenames(args.extract_folder)

    # 2) 라벨을 원하는 위치로 이동
    image_target_path = f'{args.base_path}/images/{args.target_child_folder}'
    label_target_path = f'{args.base_path}/labels/{args.target_child_folder}'
    for search_child_folder in args.search_child_folders:
        for image_filename in image_filenames:
            abs_image_file_path = f'{args.base_path}/images/{search_child_folder}/{image_filename}'
            label_filename = image_filename.replace(args.image_file_extension, args.label_file_extension)
            abs_label_file_path = f'{args.base_path}/labels/{search_child_folder}/{label_filename}'

            try:
                shutil.move(abs_image_file_path, image_target_path)
            except:
                print(f"이동 실패 : {abs_image_file_path}")
            try:
                shutil.move(abs_label_file_path, label_target_path)
            except:
                print(f"이동 실패 : {abs_label_file_path}")

move_labels_corresponding_images()