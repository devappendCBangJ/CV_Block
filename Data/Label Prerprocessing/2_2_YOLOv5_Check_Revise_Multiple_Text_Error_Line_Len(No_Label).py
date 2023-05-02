# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse

# ==============================================================
# 0. 변수 정의
# ==============================================================
parser = argparse.ArgumentParser(description='2_2_YOLOv5_Check_Revise_Multiple_Text_Error_Line_Len(No_Label)')

parser.add_argument('--base-path', default='/media/hi/SK Gold P31/Capstone/GolfBall/Golfball_Near_Remove_Similar_Test_Test_Test', type=str, help='label의 오류를 탐지할 데이터셋이 모여있는 grandmother 폴더 경로 지정')
parser.add_argument('--source-parent-folders', default=['labels'], type=str, nargs='*', help='label의 오류를 탐지할 데이터셋이 모여있는 mother 폴더 지정')
parser.add_argument('--source-child-folders', default=['train', 'val', 'test'], type=str, nargs='*', help='label의 오류를 탐지할 데이터셋이 모여있는 child 폴더 지정')

parser.add_argument('--normal-line-len', default=5, type=int, help='정상적인 경우, txt파일의 1개의 line 당 문자열의 개수')
parser.add_argument('--after-label', default='0', type=str, help='txt파일의 1개 line에 class를 제외하고 bbox의 좌표만을 포함한 총 9개의 좌표가 존재할 때, label이 없는 마지막 4개 좌표의 class를 무엇으로 지정할 것인가?')

args = parser.parse_args()

# ==============================================================
# 0. 함수 정의
# ==============================================================
def get_filenames(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

# ==============================================================
# 1. 비정상 Label 추출 + 정상화
# ==============================================================
def label_len_error_check_revise():
    # --------------------------------------------------------------
    # 1) Label 불러오기 + 메모장 저장을 위한 변수 초기화
    # --------------------------------------------------------------
    for label_folder in args.source_parent_folders:
        for train_folder in args.source_child_folders:
            errors_info = {'| filename | line count |': ''}
            errors_unique_line_len = set()

            label_filenames = get_filenames(f'{args.base_path}/{label_folder}/{train_folder}')
            # --------------------------------------------------------------
            # 2) Line 한줄씩 확인 : 비정상 데이터 저장 + 비정상 데이터 수정
            # --------------------------------------------------------------
            for label_filename in label_filenames:
                label_path = f'{args.base_path}/{args.label_folder}/{train_folder}/{label_filename}'
                # (1) 비정상 데이터 저장
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    # print(f'lines : {lines}')
                    for line in lines:
                        # print(f'line : {line}')
                        line_len = len(line.split())
                        if args.normal_line_len != line_len:
                            errors_info[label_filename] = set()
                            errors_info[label_filename].add(line_len)
                            errors_unique_line_len.add(line_len)
                # (2) 비정상 데이터 수정
                if label_filename in errors_info:
                    print(f'error_label_filename : {label_filename}')
                    with open(label_path, 'w') as f:
                        # print(f'lines : {lines}')
                        for line in lines:
                            if args.normal_line_len != len(line.split()):
                                line_list = line.split()
                                f.write(' '.join(line_list[:args.normal_line_len]))
                                f.write('\n')
                                f.write(args.after_label + ' ' + ' '.join(line_list[args.normal_line_len:]))
                                f.write('\n')
                            else:
                                f.write(line)

            # print(f'errors_info : {errors_info}')
            # print(f'abnormal_line_len : {abnormal_line_len}')

            # --------------------------------------------------------------
            # 3) 에러 정보 쓰기 (error_unique_line_len + error_file_len + error_label_path + error_line_num)
            # --------------------------------------------------------------
            error_line_len_txt_save_path = f'{args.base_path}/{args.label_folder}/error_line_len_{train_folder}.txt'
            with open(error_line_len_txt_save_path, 'w') as error_txt:
                error_txt.write(f'--------------------------------------------------------------\n')
                # (1) error_unique_line_len
                error_txt.write('error_unique_line_len :')
                for error_unique_line_len in errors_unique_line_len:
                    error_txt.write(f' {error_unique_line_len}')
                error_txt.write('\n')
                # (2) error_file_len
                error_txt.write(f'error_file_len : {len(errors_info)-1}\n')
                error_txt.write(f'--------------------------------------------------------------\n')
                # (3) error_label_path + error_line_num
                for error_label_path, error_line_num in errors_info.items():
                    error_txt.write(f'{error_label_path} | {error_line_num}\n')

label_len_error_check_revise()