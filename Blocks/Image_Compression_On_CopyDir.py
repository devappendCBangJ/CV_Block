# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
from PIL import Image
import requests

# ==============================================================
# 1. 부모 폴더 경로 지정 + 복사본 생성
# ==============================================================
data_path = "/media/hi/SK Gold P31/Korean_Emotion_Movie/val"  # 뒤에 / 붙이면 안됨

data_path_cp = f"{data_path}_cp"
if not os.path.exists(data_path_cp):
    os.makedirs(data_path_cp)

# ==============================================================
# 2. 부모 폴더 경로 내의 자식 폴더 list 추출 + 복사본 생성
# ==============================================================
data_list = os.listdir(data_path)
print(f"data_list: {data_list}")
folder_list = [data for data in data_list if os.path.isdir(f"{data_path}/{data}")]
print(f"folder_list: {folder_list}")

for idx_f, f in enumerate(folder_list):
    # if f == "기쁨":
    #     continue
    dir_path = f"{data_path}/{f}"
    dir_path_cp = f"{data_path_cp}/{f}"
    if not os.path.exists(dir_path_cp):
        os.makedirs(dir_path_cp)

    # ==============================================================
    # 3. 자식 폴더 경로 내의 파일 list 추출 + 복사본 생성
    # ==============================================================
    file_list = os.listdir(dir_path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    # print(f"{f} list_jpg: {file_list_jpg}")
    print(f"[FOLDER] {idx_f+1}/{len(folder_list)}") # 시각화

    for idx_j, j in enumerate(file_list_jpg):
        try:
            file_path = f"{dir_path}/{j}"
            file_path_cp = f"{dir_path_cp}/{j[:-4]}_com.jpg"

            # ==============================================================
            # 4. 각 이미지 파일의 압축본 -> 복사본 폴더에 저장
            # ==============================================================
            with Image.open(file_path) as im:
                if j[:-4] == ".png":
                    im = im.convert("RGB")
                im.thumbnail((300, 300))  # 원본을 300 by 300 변경
                im.save(file_path_cp, quality=100)  # quality는 jpg포맷만 유효
            if idx_j % ((len(file_list_jpg) // 10)+1) == 0: # 시각화
                print(f"[FILE] {idx_j + 1}/{len(file_list_jpg)}")
        # ==============================================================
        # 5. 에러 이미지 경로 출력 + 제거
        # ==============================================================
        except:
            print(f"[ERROR] file_path: {file_path}")    # /media/hi/SK Gold P31/Korean_Emotion_Movie/train/당황/1eab9325767c0fdbe195bcd9da5542f92609fcc1ebbb52953be96718dc57d82a_┐⌐_30_┤τ╚▓_╜╟┐▄ └┌┐¼╚»░µ_20201206220820-007-012.jpg
            os.unlink(file_path)    # os.remove(file_path)

# # ==============================================================
# # 2. 이미지 해상도 압축
# # ==============================================================
# filename_40 = filename[:-4] + "_40" + ".jpg"
# with Image.open(filename) as im:
#     im = im.convert("RGB")
#     im.save(filename_40, quality=40) #quality는 jpg포맷만 유효
#
# # ==============================================================
# # - 이미지 출력
# # ==============================================================
# pil_img = Image.open(filename_40)
# pil_img.show()
#
# # ==============================================================
# # 3. 이미지 해상도 압축 -> 사이즈 압축
# # ==============================================================
# filename_40_thumbnail = filename[:-4] + "_40_thumbnail" + ".jpg"
# with Image.open(filename_40) as im:
#     im.thumbnail((300, 300))#원본을 300 by 300 변경
#     im.save(filename_40_thumbnail) #quality는 jpg포맷만 유효
#
# # ==============================================================
# # - 이미지 출력
# # ==============================================================
# pil_img = Image.open(filename_40_thumbnail)
# pil_img.show()
#
# # ==============================================================
# # 4. 이미지 사이즈 압축
# # ==============================================================
# filename_thumbnail = filename[:-4] + "_thumbnail" + filename[-4:]
# with Image.open(filename) as im:
#     im.thumbnail((300, 300))#원본을 300 by 300 변경
#     im.save(filename_thumbnail) #quality는 jpg포맷만 유효
#
# # ==============================================================
# # - 이미지 출력
# # ==============================================================
# pil_img = Image.open(filename_thumbnail)
# pil_img.show()
#
# # ==============================================================
# # 5. 이미지 해상도 압축 + 사이즈 압축
# # ==============================================================
# filename_40_thumbnail_all = filename[:-4] + "_40_thumbnail_all" + ".jpg"
# with Image.open(filename) as im:
#     im = im.convert("RGB")
#     im.thumbnail((300, 300))  # 원본을 300 by 300 변경
#     im.save(filename_40_thumbnail_all, quality=40)  # quality는 jpg포맷만 유효
#
# # ==============================================================
# # - 이미지 출력
# # ==============================================================
# pil_img = Image.open(filename_40_thumbnail_all)
# pil_img.show()
