# transform은 사용자가 임의로 설정해주어야함!!!

# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import argparse
import numpy as np
import PIL
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# ==============================================================
# 1. 변수 선언
# ==============================================================
parser = argparse.ArgumentParser(description='1_Plot_MultipleImages')

parser.add_argument('--dataset-path', default='/home/hi/PycharmProjects/Test/Test5', type=str, help='Plot할 데이터셋이 모여있는 mother 폴더 경로 지정')
parser.add_argument('--subplot-rows', default=4, type=int, help='Plot의 Row에 들어갈 이미지 개수')
parser.add_argument('--subplot-columns', default=4, type=int, help='Plot의 Column에 들어갈 이미지 개수')

args = parser.parse_args()

# ==============================================================
# 2. 함수 정의
# ==============================================================
# 1) PIL <-> Numpy <-> Tensor 함수 정의
def pil_to_tensor(pil_image):
    """
    PIL: [width, height]
    -> NumPy: [width, height, channel]
    -> Tensor: [channel, width, height]
    """
    return torch.as_tensor(np.asarray(pil_image)).permute(2,0,1)

def tensor_to_pil(tensor_image):
    return to_pil_image(tensor_image)

def tensor_to_pltimg(tensor_image):
    """
    PIL: [width, height]
    -> NumPy: [width, height, channel]
    -> Tensor: [channel, width, height]
    """
    return tensor_image.permute(1,2,0).numpy()

# ==============================================================
# 2. 초기 세팅
# ==============================================================
# --------------------------------------------------------------
# 1) Image
# --------------------------------------------------------------
# (1) Dataset Path에서 Image List 불러오기
image_filenames = os.listdir(args.dataset_path)

# (2) Image Transform 정의
transform = transforms.RandomHorizontalFlip(1)

# --------------------------------------------------------------
# 2) Visualization
# --------------------------------------------------------------
# (1) Plt Figure Size
plt.figure(figsize=(8, 8))

# ==============================================================
# 3. Image 하나씩 출력
# ==============================================================
for i, image_filename in enumerate(image_filenames):
    # --------------------------------------------------------------
    # 1) Image 불러오기 + 변환
    # --------------------------------------------------------------
    # (1) Image 열기
    pil_image = PIL.Image.open(f'{args.dataset_path}/{image_filename}')

    # (2) Image 변환 to Tensor
    tensor_image = pil_to_tensor(pil_image)

    # (3) Image Transform
    transformed_image = transform(tensor_image)

    # --------------------------------------------------------------
    # 2) Visualization
    # --------------------------------------------------------------
    # (1) Plt Subplot Split
    plot_idx = i % (args.subplot_rows * args.subplot_columns)
    plt.subplot(args.subplot_rows, args.subplot_columns, plot_idx + 1)

    # (2) Plt Title
    plt.title(image_filename, fontsize=7)

    # (3) Plt Label
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')


    # (4) Subplot에 Image 저장
    plt.imshow(tensor_to_pltimg(transformed_image))

    # (5) Plt Figure Axis
    plt.axis("off")

    # (6) Image 시각화
    if plot_idx == 0 and i != 0:
        plt.show()


