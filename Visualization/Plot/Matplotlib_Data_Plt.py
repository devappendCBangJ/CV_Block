# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
import PIL
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# ==============================================================
# 1. 변수 선언
# ==============================================================
dataset_path = "/home/hi/Datasets/Cifar10/train/airplane/"
subplot_rows = 3
subplot_columns = 3

# ==============================================================
# 1. 함수 정의
# ==============================================================
# 1) PIL <-> Numpy <-> Tensor 함수 정의
def pil_to_tensor(pil_image):
    # PIL: [width, height]
    # -> NumPy: [width, height, channel]
    # -> Tensor: [channel, width, height]
    return torch.as_tensor(np.asarray(pil_image)).permute(2,0,1)

def tensor_to_pil(tensor_image):
    return to_pil_image(tensor_image)

def tensor_to_pltimg(tensor_image):
    return tensor_image.permute(1,2,0).numpy()

# ==============================================================
# 2. 초기 설정
# ==============================================================
# 1) Dataset
# (1) Dataset Path에서 Image List 불러오기
file_list = os.listdir(dataset_path)

# (2) Image Transform 정의
transform = transforms.RandomHorizontalFlip()

# 2) Visualization
# (1) Plt Figure Size
plt.figure(figsize=(8, 8))

# ==============================================================
# 3. 이미지 하나씩 설정
# ==============================================================
for i, file_path in enumerate(file_list):
    # 1) Dataset
    # (1) Image Tensor Load
    pil_image = PIL.Image.open(dataset_path+file_path)

    # (2) Tensor 변환 + Image Transform
    tensor = pil_to_tensor(pil_image)
    applied_image = transform(tensor)

    # 2) Visualization
    # (1) Plt Title
    plt.title(file_path)

    # (2) Plt Label
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')

    # (2) Plt Subplot Split
    plt.subplot(subplot_rows, subplot_columns, i + 1)
    plt.xticks([])
    plt.yticks([])

    # (3) Subplot에 Image 저장
    plt.imshow(tensor_to_pltimg(applied_image))

    # (4) Plt Figure Axis
    plt.axis("off")

    if i == 8:
        break

# ==============================================================
# 4. 이미지 출력
# ==============================================================
plt.show()
