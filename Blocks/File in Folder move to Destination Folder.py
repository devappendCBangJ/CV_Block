# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import shutil

# ==============================================================
# 1. 부모 폴더 경로 지정 + 복사본 생성
# ==============================================================
data_path = "/media/hi/SK Gold P31/Korean_Emotion_Movie/compression/train"  # 뒤에 / 붙이면 안됨

# ==============================================================
# 2. 부모 폴더 경로 내의 자식 폴더 list 추출
# ==============================================================
data_list = os.listdir(data_path)
# print(f"data_list: {data_list}")
folder_list = [data for data in data_list if os.path.isdir(f"{data_path}/{data}")]
print(f"folder_list: {folder_list}")

# ==============================================================
# 3. 출발 지점(이동할) 폴더 list 추출 + 도착 지점 폴더 list 추출
# ==============================================================
mother_list = []
child_list = []
for folder in folder_list:
    if not folder[-1:].isdigit():
        mother_list.append(folder)
    else:
        child_list.append(folder)
print(f"mother_list: {mother_list}")
print(f"child_list: {child_list}")

# ==============================================================
# 4. 출발 지점(이동할) 폴더 내의 jpg list 추출 (출발 지점 폴더와, 도착 지점 폴더 제목의 끝 글자를 제외한 폴더명이 같은 경우) -> 도착 지점 폴더로 이동
# ==============================================================
for child in child_list:
    child_path = data_path + "/" + child
    for mother in mother_list:
        mother_path = data_path + "/" + mother
        if child[:-1] == mother:
            print(f"child, mother : {child, mother}")

            child_file_list = os.listdir(child_path)
            child_file_list_jpg = [child_file for child_file in child_file_list if child_file.endswith(".jpg")]
            # print(f"file_list_jpg : {child_file_list_jpg}")

            for idx, child_file_path in enumerate(child_file_list_jpg):
                try:
                    child_file_path_full = child_path + "/" + child_file_path
                    # print(f"child_file_path_full, mother_path : {child_file_path_full}, {mother_path}")
                    shutil.move(child_file_path_full, mother_path)

                    if idx % ((len(child_file_list_jpg) // 10)+1) == 0: # 시각화
                        print(f"[FILE] {idx + 1}/{len(child_file_list_jpg)}")
                except:
                    print(f"[path] child_file_path_full, mother_path : {child_file_path_full}, {mother_path}")
