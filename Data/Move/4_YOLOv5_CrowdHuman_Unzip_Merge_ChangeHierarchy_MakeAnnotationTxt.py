"""
Based on https://raw.githubusercontent.com/jkjung-avt/yolov4_crowdhuman/master/data/gen_txts.py

Inputs:
    * nothing
    * or folder with CrowdHuman_train01.zip, CrowdHuman_train02.zip, CrowdHuman_train03.zip, CrowdHuman_val.zip, annotation_train.odgt, annotation_val.odgt

python crowdhuman_to_yolo.py --dataset_path foo/bar/

Outputs:
    * same folder with :
        - labels/train
        - labels/val
        - images/train
        - images/val
"""

# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import json
import os
import shutil
import zipfile
from pathlib import Path
from argparse import ArgumentParser
import requests

import cv2

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = ArgumentParser()
parser.add_argument('--dataset-path', default='/media/hi/SK Gold P31/Capstone/CrowdHuman', type=str, help='Dataset name')
parser.add_argument('--skip-download', action='store_true', help='Download the dataset from GoogleDrive (may fail)')

args = parser.parse_args()

# --------------------------------------------------------------
# 1. Main문
    # 2) 데이터셋 다운로드 + 폴더 합치기
        # (4) train / val 폴더 합치기
            # 1] train 목적지 폴더 생성
# --------------------------------------------------------------
def make_dir_ignore(path):
    # [1] 시도 : 폴더 생성
    try:
        os.makedirs(path)
    # [2] 예외 : print로 알려줌
    except:
        print("")

# --------------------------------------------------------------
# 안써도 됨
# --------------------------------------------------------------
def download_file_from_google_drive(id, destination):
    # https://stackoverflow.com/a/39225039/7036639
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

# --------------------------------------------------------------
# 2) 데이터셋 다운로드 + 폴더 합치기
# --------------------------------------------------------------
def download_crowd_dataset(dataset_abs_path, skip_download=True):
    # (1) 데이터셋 경로 불러오기
    dataset_abs_path = str(dataset_abs_path)
    # (2) 데이터셋별로 url 정의
    dataset_url_dict = {"CrowdHuman_train01.zip": "134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y",
                      "CrowdHuman_train02.zip": "17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla",
                      "CrowdHuman_train03.zip": "1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW",
                      "CrowdHuman_val.zip": "18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO",
                      "annotation_train.odgt": "1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3",
                      "annotation_val.odgt": "10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL"
                      }
    # (3) 데이터셋 zip파일 다운로드 + 압축 해제
    for zip_filename, drive_file_id in dataset_url_dict.items():
        zip_filename_abs_path = os.path.join(dataset_abs_path, zip_filename)
        # 1] 다운로드 건너띄기 설정한 경우 or 파일 존재하는 경우 -> 구글드라이브 url에서 파일 다운로드
        if not skip_download and not os.path.isfile(zip_filename_abs_path):
            print("File not found, trying to download it...")
            download_file_from_google_drive(drive_file_id, zip_filename_abs_path)
        # 2] zip 파일인 경우 -> 이미지 압축 해제
        if ".zip" in zip_filename:
            unzipped_filename = zip_filename_abs_path.replace(".zip", '')
            with zipfile.ZipFile(zip_filename_abs_path, "r") as zip_ref:
                zip_ref.extractall(unzipped_filename)
    # (4) train / val 폴더 합치기
    # 1] train 목적지 폴더 생성
    dest_images_train_dir_abs_path = os.path.join(dataset_abs_path, "images", "train")
    make_dir_ignore(dest_images_train_dir_abs_path)
    # 2] train 폴더 합치기
    for train_folder in ["CrowdHuman_train01", "CrowdHuman_train02", "CrowdHuman_train03"]:
        src_images_train_dir_abs_path = os.path.join(dataset_abs_path, train_folder, "Images")
        src_imagenames = os.listdir(src_images_train_dir_abs_path)
        for f in src_imagenames:
            shutil.move(os.path.join(src_images_train_dir_abs_path, f), dest_images_train_dir_abs_path)
        shutil.rmtree(os.path.join(dataset_abs_path, train_folder))
    # 3] val 목적지 폴더 생성
    dest_images_val_dir_abs_path = os.path.join(dataset_abs_path, "images", "val")
    make_dir_ignore(dest_images_val_dir_abs_path)
    # 4] val 폴더 합치기
    val_folder = os.path.join(dataset_abs_path, "CrowdHuman_val")
    src_images_val_dir_abs_path = os.path.join(val_folder, "Images")
    src_imagenames = os.listdir(src_images_val_dir_abs_path)
    for f in src_imagenames:
        shutil.move(os.path.join(src_images_val_dir_abs_path, f), dest_images_val_dir_abs_path)
    shutil.rmtree(val_folder)

# --------------------------------------------------------------
# 1. Main문
    # 4) annotation txt 파일 생성 for YOLOv5
        # (2) annotation 파일 열기
            # 3] 이미지 shape 추출 (이미지 절대 경로 기반)
# --------------------------------------------------------------
def image_shape(ID, image_abs_path):
    assert image_abs_path is not None
    jpg_path = image_abs_path / ('%s.jpg' % ID)
    # print(jpg_path)
    img = cv2.imread(jpg_path.as_posix())
    return img.shape

# --------------------------------------------------------------
# 1. Main문
    # 4)
        # (2) annotation 파일 열기
            # 4] txt 파일 생성 + 쓰기
                # [1] gtboxes 순회
                    # 2]] 태그가 person인 경우
                        # [[1]] hbox 좌표 쓰기 in txt
                        # [[2]] fbox 좌표 쓰기 in txt
# --------------------------------------------------------------
def txt_line(cls, bbox, img_w, img_h):
    # 1. bbox 좌표 불러오기
    x, y, w, h = bbox
    # 2. x, y, w, h 추출 : x, y의 좌표가 -인 경우 0으로 바꿔줌 / w, h좌표가 이미지 실제 w, h보다 큰 경우 이미지 최대 픽셀로 바꿔줌
    x = max(int(x), 0)
    y = max(int(y), 0)
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)
    # 3. cx, cy, nw, nh 추출 : 정수 좌표 -> 비율 좌표 변환
    cx = (x + w / 2.) / img_w
    cy = (y + h / 2.) / img_h
    nw = float(w) / img_w
    nh = float(h) / img_h
    # 4. class, cx, cy, nw, nh 반환
    return '%d %.6f %.6f %.6f %.6f\n' % (cls, cx, cy, nw, nh)

# --------------------------------------------------------------
# 1. Main문
    # 4) annotation txt 파일 생성 for YOLOv5
# --------------------------------------------------------------
def make_annotation_txt(train_val_dir='val', annotation_filename='annotation_val.odgt', dataset_abs_path=None):
    # (1) labels/val or labels/train 폴더 생성
    assert dataset_abs_path is not None
    dataset_abs_path.mkdir(exist_ok=True)
    images_abs_path = []
    make_dir_ignore(dataset_abs_path / "labels" / train_val_dir)

    # (2) annotation 파일 열기
    with open(annotation_filename, 'r') as anno_files:
        # 1] annotation 한줄씩 읽기
        for anno_lines in anno_files.readlines():
            anno_line = json.loads(anno_lines)
            # 2] annotation ID 추출
            ID = anno_line['ID']  # e.g. '273271,c9db000d5146c15'
            # print('Processing ID: %s' % ID)
            # 3] 이미지 shape 추출 (이미지 3채널 제한)
            img_h, img_w, img_c = image_shape(ID, dataset_abs_path / Path("images") / Path(train_val_dir))
            assert img_c == 3  # should be a BGR image
            # 4] txt 파일 생성 + 쓰기
            txt_abs_path = dataset_abs_path / "labels" / train_val_dir / ('%s.txt' % ID)
            with open(txt_abs_path.as_posix(), 'w') as txt_file:
                # [1] gtboxes 순회
                for anno_obj in anno_line['gtboxes']:
                    # 1]] 태그가 mask인 경우 (person이 아닌 경우) : 건너띄기
                    if anno_obj['tag'] == 'mask':
                        continue  # ignore non-human
                    # 2]] 태그가 person인 경우
                    assert anno_obj['tag'] == 'person'
                    # [[1]] 얼굴 좌표(head box) 쓰기 in txt
                    if 'hbox' in anno_obj.keys():  # head
                        line = txt_line(1, anno_obj['hbox'], img_w, img_h)
                        if line:
                            txt_file.write(line)
                    # [[2]] 전체 몸(full box) 좌표 쓰기 in txt
                    if 'fbox' in anno_obj.keys():  # full body
                        line = txt_line(0, anno_obj['fbox'], img_w, img_h)
                        if line:
                            txt_file.write(line)
            # 5] 분류 완료한 이미지 경로 저장 in [train.txt or val.txt]
            images_abs_path.append('%s/%s.jpg' % (dataset_abs_path / Path("images") / Path(train_val_dir), ID))
    # write the 'data/crowdhuman/train.txt' or 'data/crowdhuman/test.txt'
    set_txt_abs_path = dataset_abs_path / ('%s.txt' % train_val_dir)
    with open(set_txt_abs_path.as_posix(), 'w') as set_txt:
        for image_abs_path in images_abs_path:
            set_txt.write('%s\n' % image_abs_path)

# --------------------------------------------------------------
# 1. Main문
    # 3) txt 파일 제거 in dataset_abs_path
# --------------------------------------------------------------
def rm_txts(dataset_abs_path):
    # (1) txt 파일 제거 in dataset_abs_path
    for txt in dataset_abs_path.glob('*.txt'):
        if txt.is_file():
            txt.unlink()

# ==============================================================
# 1. Main문
# ==============================================================
def main():
    print(args)
    # 1) 데이터셋 경로 불러오기
    dataset_abs_path = Path(args.dataset_path)
    if not dataset_abs_path.is_dir():
        os.mkdir(dataset_abs_path)
    """
    # 2) 데이터셋 다운로드 + 폴더 합치기 in dataset_abs_path
    download_crowd_dataset(dataset_abs_path, skip_download=(not args.skip_download))
    # 3) txt 파일 제거 in dataset_abs_path
    rm_txts(dataset_abs_path)
    """
    # 4) annotation txt 파일 생성 for YOLOv5
    make_annotation_txt('val', dataset_abs_path / Path('annotation_val.odgt'), dataset_abs_path)
    make_annotation_txt('train', dataset_abs_path / Path('annotation_train.odgt'), dataset_abs_path)

if __name__ == '__main__':
    main()
