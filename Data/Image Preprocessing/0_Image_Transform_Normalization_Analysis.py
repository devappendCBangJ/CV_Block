# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# ==============================================================
# 0. 함수 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 이미지의 RGB 채널별 통계량 확인
# --------------------------------------------------------------
def print_stats(dataset):
    imgs = np.array([img.numpy() for img, _ in dataset])
    print(f'shape: {imgs.shape}')

    min_r = np.min(imgs, axis=(2, 3))[:, 0].min()
    min_g = np.min(imgs, axis=(2, 3))[:, 1].min()
    min_b = np.min(imgs, axis=(2, 3))[:, 2].min()

    max_r = np.max(imgs, axis=(2, 3))[:, 0].max()
    max_g = np.max(imgs, axis=(2, 3))[:, 1].max()
    max_b = np.max(imgs, axis=(2, 3))[:, 2].max()

    mean_r = np.mean(imgs, axis=(2, 3))[:, 0].mean()
    mean_g = np.mean(imgs, axis=(2, 3))[:, 1].mean()
    mean_b = np.mean(imgs, axis=(2, 3))[:, 2].mean()

    std_r = np.std(imgs, axis=(2, 3))[:, 0].std()
    std_g = np.std(imgs, axis=(2, 3))[:, 1].std()
    std_b = np.std(imgs, axis=(2, 3))[:, 2].std()

    print(f'min: {min_r, min_g, min_b}')
    print(f'max: {max_r, max_g, max_b}')
    print(f'mean: {mean_r, mean_g, mean_b}')
    print(f'std: {std_r, std_g, std_b}')

# --------------------------------------------------------------
# 2) 이미지의 평균 + 표준편차 확인
# --------------------------------------------------------------
def calculate_norm(dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

# ==============================================================
# 1. Main문
# ==============================================================
# --------------------------------------------------------------
# 1) 데이터셋 Tensor 변환
# --------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

# --------------------------------------------------------------
# 2) 데이터셋 불러오기
# --------------------------------------------------------------
train = datasets.CIFAR10(root='data',
                         train=True,
                         download=True,
                         transform=transform
                        )

test = datasets.CIFAR10(root='data',
                        train=False,
                        download=True,
                        transform=transform
                       )

# --------------------------------------------------------------
# 3) 데이터의 평균, 표준편차 계산
# --------------------------------------------------------------
mean_, std_ = calculate_norm(train)
print('==='*10)
print(f'평균(R,G,B): {mean_}\n표준편차(R,G,B): {std_}')
print('==='*10)

# --------------------------------------------------------------
# 4) 데이터셋 Tensor 변환 + 정규화
# --------------------------------------------------------------
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -1 ~ 1 사이의 범위를 가지도록 정규화
# ])
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_, std_),
])

# --------------------------------------------------------------
# 5) 정규화된 데이터셋 불러오기
# --------------------------------------------------------------
train = datasets.CIFAR10(root='data',
                         train=True,
                         download=True,
                         transform=transform
                        )

test = datasets.CIFAR10(root='data',
                        train=False,
                        download=True,
                        transform=transform
                       )

# --------------------------------------------------------------
# 6) 결과 출력
# --------------------------------------------------------------
print_stats(train)
print('==='*10)
print_stats(test)
print('==='*10)