# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import torch
import torch.nn as nn

# ==============================================================
# 1. X, Y 생성
# ==============================================================
x_data = torch.Tensor([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
])

y_data = torch.LongTensor([
    0,  # etc
    1,  # mammal
    2,  # birds
    0,
    0,
    2
])

# ==============================================================
# 2. Model 정의
# ==============================================================
class DNN2(nn.Module):
    def __init__(self):
        super(DNN2, self).__init__()
        self.w1 = nn.Linear(2, 10)
        self.bias1 = torch.zeros([10])

        self.w2 = nn.Linear(10, 3)
        self.bias2 = torch.zeros([3])

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        y = self.w1(x) + self.bias1
        y = self.relu(y)

        y = self.w2(y) + self.bias2
        return y

class DNN3(nn.Module):
    def __init__(self):
        super(DNN3, self).__init__()
        self.w1 = nn.Linear(2, 10)
        self.bias1 = torch.zeros([10])

        self.w2 = nn.Linear(10, 3)
        self.bias2 = torch.zeros([3])

        self.w3 = nn.Linear(3, 1)
        self.bias3 = torch.zeros([1])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        y = self.w1(x) + self.bias1
        y = self.relu(y)

        y = self.w2(y) + self.bias2
        y = self.relu(y)

        y = self.w3(y) + self.bias3
        return y

# ==============================================================
# 3. Model 선언
# ==============================================================
# 1) Model
model2 = DNN2()
# 2) Loss
criterion = torch.nn.CrossEntropyLoss()
# 3) Optimizer
optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)

# ==============================================================
# 4. Load Model
# ==============================================================
# 1) Save Path
Save_Path = './weights/'

# # --------------------------------------------------------------
# # 2-1) Load Model
# # --------------------------------------------------------------
# model2 = torch.load(Save_Path + 'model2.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
#
# # --------------------------------------------------------------
# # 2-2) Load Model StateDict
# # --------------------------------------------------------------
# model2.load_state_dict(torch.load(Save_Path + 'model2_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

# # --------------------------------------------------------------
# # 2-3) Save Model & Other StateDict
# # --------------------------------------------------------------
# checkpoint = torch.load(Save_Path + 'all_state_dict2.tar')   # dict 불러오기
# model2.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])

# --------------------------------------------------------------
# 2-4) Transfer Learning (Load Model StateDict)
# --------------------------------------------------------------
model2.load_state_dict(torch.load(Save_Path + 'model3_state_dict.pt'), strict=False)
