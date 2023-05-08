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
parser = argparse.ArgumentParser(description='1_Save_Filenames')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_Test', type=str, help='라벨과 이미지 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-pathes', default=['images'], type=str, nargs='*', help='source 폴더 기준 부모 폴더들 경로')
parser.add_argument('--source-child-pathes', default=['train'], type=str, nargs='*', help='source 폴더 기준 자식 폴더들 경로')

parser.add_argument('--before-file-extension', default='.jpg', type=str, help='base_path/label_folder/source_child_folders 안에 들어있는 텍스트 파일 확장자')
parser.add_argument('--after-file-extension', default='.txt', type=str, help='base_path/after_folder/source_child_folders 에서 오픈할 이미지 파일 확장자')

args = parser.parse_args()

# ==============================================================
# 1. 파일명 추출
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# ==============================================================
# 2. 파일명 저장
# ==============================================================
image_filenames = []

def save_filenames():
    for image_path in args.source_parent_pathes:
        for train_path in args.source_child_pathes:
            temp_filenames = get_filenames(f'{args.base_path}/{image_path}/{train_path}')
            for temp_filename in temp_filenames:
                image_filenames.append(f'{args.base_path}/{image_path}/{train_path}/{temp_filename}')

    target_images_save_path = f'{args.base_path}/target_images.txt'
    with open(target_images_save_path, 'w') as txt:
        for image_filename in image_filenames:
            txt.write(f'{image_filename}\n')

save_filenames()