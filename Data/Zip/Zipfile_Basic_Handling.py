# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import zipfile
import os
import pandas as pd

# ==============================================================
# 1. 경로 설정
# ==============================================================
os.chdir("/home/hi/PycharmProjects/Test/zipfile/") # 이렇게 하지 않으면 상위 경로 전체가 한꺼번에 압축되어버림 ★★★

# ==============================================================
# 2. 압축
# ==============================================================
# 1) 특정파일 압축
my_zip = zipfile.ZipFile("0.zip", 'w')

my_zip.write('test1.txt')
my_zip.close()

# 2) 여러파일 압축
file_ls = ['test1.txt', 'test2.txt', 'test3.txt', 'testtest.csv']
with zipfile.ZipFile("test_zip.zip", 'w') as my_zip:
    for i in file_ls:
        my_zip.write(i)
    my_zip.close()

# ==============================================================
# 3. 압축 해제
# ==============================================================
# 1) 특정파일 압축 해제
zipfile.ZipFile('test_zip.zip').extract('test1.txt')

# 2) 모든파일 압축 해제
zipfile.ZipFile('test_zip.zip').extractall()

# ==============================================================
# 4. 압축 파일 읽기
# ==============================================================
# 1) 특정파일 읽기
my_zip.read('testtest.csv')
print(pd.read_csv('testtest.csv', sep=','))

# 2) 압축파일 내 파일명 읽기
my_zip.namelist()

# 3) 압축파일 정보 확인
zp_info = my_zip.getinfo('testtest.csv')   # csv파일의 Zipinfo객체 생성
print(zp_info.filename)                    # 파일명
print(zp_info.file_size)                   # 파일용량
print(zp_info.date_time)                   # 작성일자
print(zp_info.compress_size)
print(zp_info.comment)                     # 주석문