# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import argparse
import json
import os
import shutil

import cv2

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='0_2_JSON2YOLO_Copy_Move_Corresponding_Image')

parser.add_argument('--json-mother-abs-path', default='/media/hi/SK Gold P31/Capstone/COCO/annotations', type=str, help='json 파일이 모여있는 폴더 지정')
parser.add_argument('--json-filenames', default=['instances_train2017.json', 'instances_val2017.json'], type=str, help='yolo로 변환할 json 파일명 지정 (단, json 파일명에 train, val, test 중 해당하는 단어를 포함하고 있어야함)')
parser.add_argument('--val-image-mother-abs-path', default='/media/hi/SK Gold P31/Capstone/COCO/val2017', type=str, help='val 이미지들이 모여있는 폴더 지정')
parser.add_argument('--train-image-mother-abs-path', default='/media/hi/SK Gold P31/Capstone/COCO/train2017', type=str, help='train 이미지들이 모여있는 폴더 지정')
parser.add_argument('--test-image-mother-abs-path', default='/media/hi/SK Gold P31/Capstone/COCO/test2017', type=str, help='test 이미지들이 모여있는 폴더 지정')
parser.add_argument('--target-cls', default=['37'], type=str, help='yolo 파일로 변환시킬 class 선택, 모든 class를 변환하고 싶은 경우 All 이라고 지정하면 됨')

args = parser.parse_args()

"""
# ==============================================================
# 2. 폴더 내 파일 합치기
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames
"""

# ==============================================================
# 1. Main문
# ==============================================================
def main():
    print(f'args : {args}')
    # --------------------------------------------------------------
    # 1) labels, images 각각 YOLO 폴더 생성
    # --------------------------------------------------------------
    if not os.path.exists(f"{args.json_mother_abs_path}/labels"):
        os.makedirs(f"{args.json_mother_abs_path}/labels")
        os.makedirs(f"{args.json_mother_abs_path}/images")

    # --------------------------------------------------------------
    # 2) labes, images 각각 train_val_dir 폴더 생성
    # --------------------------------------------------------------
    for train_val_dir in ['train', 'val', 'test']:
        if not os.path.exists(f"{args.json_mother_abs_path}/labels/{train_val_dir}"):
            os.makedirs(f"{args.json_mother_abs_path}/labels/{train_val_dir}")
            os.makedirs(f"{args.json_mother_abs_path}/images/{train_val_dir}")

    """
    # --------------------------------------------------------------
    # 3) train, val, test 내 모든 이미지 set으로 만들기
    # --------------------------------------------------------------
    train_images_filenames = set(get_filenames(args.train_image_mother_abs_path))
    val_images_filenames = set(get_filenames(args.val_image_mother_abs_path))
    test_images_filenames = set(get_filenames(args.test_image_mother_abs_path))
    """

    # --------------------------------------------------------------
    # 4) json 파일 1개씩 읽음
    # --------------------------------------------------------------
    for json_filename in args.json_filenames:
        images_abs_path = []
        errors_info = ['| filename | cls | x | y | img_w | img_h |']

        with open(f"{args.json_mother_abs_path}/{json_filename}", 'r') as json_abs_path:
            # --------------------------------------------------------------
            # (1) json 대상 종류 추출 (JSON 파일명에 train, val, test 중 하나의 단어가 존재해야함)
            # --------------------------------------------------------------
            if "val" in json_filename:
                train_val_dir = 'val'
            elif "train" in json_filename:
                train_val_dir = 'train'
            elif "test" in json_filename:
                train_val_dir = 'test'
            else:
                raise Exception("Error:JSON 파일명에 train, val, test 중 하나의 단어가 존재하지 않습니다.")

            # --------------------------------------------------------------
            # (2) json 파일 불러오기
            # --------------------------------------------------------------
            json_data = json.load(json_abs_path)

            # --------------------------------------------------------------
            # (3) json 파일의 모든 annotation 중, target label에 해당하는 label만 yolo 형식으로 변환
            # --------------------------------------------------------------
            for j in range(len(json_data["annotations"])):
                # 1] 클래스명 추출
                cls = json_data["annotations"][j]["category_id"]
                if (str(cls) in args.target_cls) or args.target_cls == "All":
                    # 2] 이미지명 추출
                    image_filename = json_data["annotations"][j]["image_id"]
                    image_filename = (str(image_filename)).zfill(12)

                    # 3] 타겟 텍스트 + source 이미지 경로 (image_mother 폴더 경로명에 train, val, test 중 하나의 단어가 존재해야함) + target image 경로
                    txt_abs_path = f"{args.json_mother_abs_path}/labels/{train_val_dir}/{image_filename}.txt"
                    if train_val_dir in args.val_image_mother_abs_path:
                        image_abs_path = f"{args.val_image_mother_abs_path}/{image_filename}.jpg"
                    elif train_val_dir in args.train_image_mother_abs_path:
                        image_abs_path = f"{args.train_image_mother_abs_path}/{image_filename}.jpg"
                    elif train_val_dir in args.test_image_mother_abs_path:
                        image_abs_path = f"{args.test_image_mother_abs_path}/{image_filename}.jpg"
                    else:
                        raise Exception("Error:image_mother 폴더 경로명에 train, val, test 중 하나의 단어가 존재하지 않습니다.")

                    target_image_folder_abs_path = f"{args.json_mother_abs_path}/images/{train_val_dir}"

                    # 4] 이미지 img_x, img_y 추출 + 이미지 정보 추출
                    image = cv2.imread(image_abs_path)
                    img_h, img_w, img_c = image.shape

                    # 5] x, y, w, h 추출 & 변환
                        # [1] x, y의 좌표가 이미지 최소 좌표보다 작은 경우, 0으로 바꿔줌
                        # [2] x, y 좌표가 실제 w, h 좌표보다 큰 경우, 제거
                        # [3] w, h좌표가 이미지 실제 w, h보다 큰 경우 이미지 최대 픽셀로 바꿔줌
                    x, y, w, h = json_data["annotations"][j]["bbox"]

                    x = max(int(x), 0)
                    y = max(int(y), 0)

                    if x >= img_w or y >= img_h:
                        print(f'errors_info : {txt_abs_path} {cls} {x} {y} {img_w} {img_h}')
                        errors_info.append(f'{txt_abs_path} {cls} {x} {y} {img_w} {img_h}')

                    w = min(int(w), img_w - x)
                    h = min(int(h), img_h - y)

                    # 6] cx, cy, nw, nh 변환 : 정수 좌표 -> 비율 좌표 변환
                    cx = (x + w / 2.) / img_w
                    cy = (y + h / 2.) / img_h
                    nw = float(w) / img_w
                    nh = float(h) / img_h

                    line = '%d %.6f %.6f %.6f %.6f\n' % (cls, cx, cy, nw, nh)

                    # 7] yolo 형식 변환 파일 저장
                    with open(txt_abs_path, 'a') as txt_file:
                        txt_file.write(line)

                    # 8] yolo 형식에 맞게 이미지 파일 복사
                    shutil.copy(image_abs_path, target_image_folder_abs_path)

                    # 9] 분류 완료한 이미지 경로 모아두기
                    images_abs_path.append(image_abs_path)

                # 9] 출력 : yolo 변환 완료 annotation 개수
                if j % ((len(json_data["annotations"]) // 50)+1) == 0:
                    print(f'annotation_count : {j}/{len(json_data["annotations"])}')

            # --------------------------------------------------------------
            # (4) 분류 완료한 이미지 경로 저장 (중복 제거)
            # --------------------------------------------------------------
            images_abs_path = set(images_abs_path)

            set_txt_save_abs_path = f"{args.json_mother_abs_path}/labels/{train_val_dir}.txt"
            with open(set_txt_save_abs_path, 'w') as set_txt:
                for image_abs_path in images_abs_path:
                    set_txt.write('%s\n' % image_abs_path)

            # --------------------------------------------------------------
            # (5) 분류 완료한 이미지 경로 저장 (중복 제거)
            # --------------------------------------------------------------
            errors_info = set(errors_info)

            error_txt_save_abs_path = f"{args.json_mother_abs_path}/labels/error_{train_val_dir}.txt"
            with open(error_txt_save_abs_path, 'w') as error_txt:
                for error_info in errors_info:
                    error_txt.write('%s\n' % error_info)

            # --------------------------------------------------------------
            # (6) 출력 : yolo 변환 완료 이미지 개수
            # --------------------------------------------------------------
            print(f'image_count : {len(images_abs_path)}')

# ==============================================================
# 1. Main문
# ==============================================================
if __name__ == '__main__':
    main()
