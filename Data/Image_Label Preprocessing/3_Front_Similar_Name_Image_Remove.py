# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='3_Front_Similar_Name_Image_Remove')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_New', type=str, help='데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--filename-split-criterions', default=['_jpeg', '_JPEG', '_png', '_jpg'], type=str, nargs='*', help='이미지명 Split 기준')

parser.add_argument('--source-parent-pathes', default=['images'], type=str, nargs='*', help='데이터셋의 이미지가 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-pathes', default=['train', 'val', 'test'], type=str, nargs='*', help='데이터셋의 이미지가 모여있는 child 폴더 지정')

parser.add_argument('--image-folder', default='images', type=str, help='base_path/source_parent_pathes/source_child_pathes에서 이미지가 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--label-folder', default='labels', type=str, help='base_path/source_parent_pathes/source_child_pathes에서 라벨이 모여있는 source_parent_pathes 폴더명')
parser.add_argument('--before-file-extension', default='.jpg', type=str, help='base_path/image_folder/source_child_pathes 안에 들어있는 이미지 파일 확장자')
parser.add_argument('--after-file-extension', default='.txt', type=str, help='base_path/label_folder/source_child_pathes 에서 오픈할 텍스트 파일 확장자')

args = parser.parse_args()

def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

def Remove_Image():
    # --------------------------------------------------------------
    # 1) Image File list 불러오기
    # --------------------------------------------------------------
    for image_path in args.source_parent_pathes:
        for train_path in args.source_child_pathes:
            target_images = dict()
            splited_image_filenames = []

            image_filenames = get_filenames(f'{args.base_path}/{image_path}/{train_path}')
            image_filenames.sort()

            # --------------------------------------------------------------
            # (1) Split 기준에 부합하는 Image File의 앞부분만 저장 + Sort (_jpg, _png 등이 동시에 존재하는 경우 있으므로, 동일한 파일을 중복해서 list에 넣는 것을 방지하기 위해 break)
            # --------------------------------------------------------------
            for i, image_filename in enumerate(image_filenames):
                for filename_split_criterion in args.filename_split_criterions:
                    image_split_idx = image_filename.find(filename_split_criterion)
                    if image_split_idx != -1:
                        splited_image_filenames.append([i, image_filename[0:image_split_idx]])
                        break

            """
            print("image_filenames : ", image_filenames)
            """

            print(len(image_filenames), len(splited_image_filenames))

            # --------------------------------------------------------------
            # (2) Splited Image File 순회하면서, 다음 파일명과 같으면 사진 용량 저장, 다음 파일명과 다르면 용량 최대인 사진만 남기고 중복 파일 삭제 / 마지막 순회인 경우 추가적으로, 용량 최대인 사진만 남기고 중복 파일 삭제
            # --------------------------------------------------------------
            for i in range(len(splited_image_filenames)-1):
                # --------------------------------------------------------------
                # 1] Split된 현재 이미지명 == Split된 다음 이미지명인 경우 : 현재, 다음 파일 용량 저장
                # --------------------------------------------------------------
                if splited_image_filenames[i][1] == splited_image_filenames[i+1][1]:
                    # [1] buffer 비어있는 경우 : 현재, 다음 파일 용량 저장
                    if len(target_images) == 0:
                        now_image_idx, next_image_idx = splited_image_filenames[i][0], splited_image_filenames[i+1][0]
                        now_image_filename, next_image_filename = image_filenames[now_image_idx], image_filenames[next_image_idx]
                        now_image_path, next_image_path = f'{args.base_path}/{args.image_folder}/{train_path}/{now_image_filename}', f'{args.base_path}/{args.image_folder}/{train_path}/{next_image_filename}'
                        now_image_filesize, next_image_filesize = os.path.getsize(now_image_path), os.path.getsize(next_image_path)
                        target_images[splited_image_filenames[i][0]] = now_image_filesize
                        target_images[splited_image_filenames[i+1][0]] = next_image_filesize
                    # [2] buffer 1개 존재하는 경우 : 오류
                    elif len(target_images) == 1:
                        raise Exception("target_images 내에 원소가 1개만 존재합니다!")
                    # [3] buffer 2개 이상 존재하는 경우 : 다음 파일 용량 저장
                    else:
                        next_image_idx = splited_image_filenames[i+1][0]
                        next_image_filename = image_filenames[next_image_idx]
                        next_image_path = f'{args.base_path}/{args.image_folder}/{train_path}/{next_image_filename}'
                        next_image_filesize = os.path.getsize(next_image_path)
                        target_images[splited_image_filenames[i+1][0]] = next_image_filesize
                # --------------------------------------------------------------
                # 2] Split된 현재 이미지명 != Split된 다음 이미지명인 경우 : 파일 용량 최대값인 파일 빼고, 중복되는 파일 삭제
                # --------------------------------------------------------------
                else:
                    # [1] buffer 비어있는 경우 : pass
                    if len(target_images) == 0:
                        pass
                    # [2] buffer 1개 존재하는 경우 : 오류
                    elif len(target_images) == 1:
                        raise Exception("target_images 내에 원소가 1개만 존재합니다!")
                    # [3] buffer 2개 이상 존재하는 경우 : 파일 용량 최대값인 파일 빼고, 중복되는 파일 삭제
                    else:
                        target_images.pop(max(target_images, key=target_images.get))
                        for remove_image_idx in target_images.keys():
                            print(remove_image_idx)
                            print(len(image_filenames))
                            remove_image_path = f'{args.base_path}/{args.image_folder}/{train_path}/{image_filenames[remove_image_idx]}'
                            remove_label_path = f'{args.base_path}/{args.label_folder}/{train_path}/{image_filenames[remove_image_idx].replace(args.before_file_extension, args.after_file_extension)}'

                            print(f"remove_image_path : {remove_image_path}")
                            print(f"remove_label_path : {remove_label_path}")

                            os.unlink(remove_image_path)
                            os.unlink(remove_label_path)

                        # list 비우기
                        target_images = dict()
                # --------------------------------------------------------------
                # 3] 마지막 순회인 경우 : 파일 용량 최대값인 파일 빼고, 중복되는 파일 삭제
                # --------------------------------------------------------------
                if i == len(splited_image_filenames)-1:
                    # [1] buffer 2개 이상 존재하는 경우 : 파일 용량 최대값인 파일 빼고, 중복되는 파일 삭제
                    if len(target_images) >= 2:
                        target_images.pop(max(target_images, key=target_images.get))
                        for remove_image_idx in target_images.keys():
                            remove_image_path = f'{args.base_path}/{args.image_folder}/{train_path}/{image_filenames[remove_image_idx]}'
                            remove_label_path = f'{args.base_path}/{args.label_folder}/{train_path}/{image_filenames[remove_image_idx].replace(args.before_file_extension, args.after_file_extension)}'

                            print(f"remove_image_path : {remove_image_path}")
                            print(f"remove_label_path : {remove_label_path}")

                            os.unlink(remove_image_path)
                            os.unlink(remove_label_path)

Remove_Image()