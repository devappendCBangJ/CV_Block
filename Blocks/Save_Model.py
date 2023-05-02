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

        self.w3 = nn.Linear(3, 10)
        self.bias3 = torch.zeros([10])

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
model3 = DNN3()
# 2) Loss
criterion = torch.nn.CrossEntropyLoss()
# 3) Optimizer
optimizer = torch.optim.SGD(model3.parameters(), lr=0.01)
print("model3.parameters() : ", model3.parameters())

# ==============================================================
# 4. Train Model
# ==============================================================
# 1) Epoch 반복
for epoch in range(1000):
    # (1) Model Output 추출
    output = model3(x_data)
    # (2) Loss 계산
    loss = criterion(output, y_data)

    # (3) Loss Update
    optimizer.zero_grad()
    loss.backward()
    # (4) Optimizer Update
    optimizer.step()

    print("progress:", epoch, "loss=", loss.item())

# ==============================================================
# 5. Save Model
# ==============================================================
# 1) Save Path
Save_Path = './weights/'

# --------------------------------------------------------------
# 2-1) Save Model
# --------------------------------------------------------------
torch.save(model3, Save_Path + 'model3.pt')  # 전체 모델 저장

# --------------------------------------------------------------
# 2-2) Save Model StateDict
# --------------------------------------------------------------
torch.save(model3.state_dict(), Save_Path + 'model3_state_dict.pt')  # 모델 객체의 state_dict 저장

# --------------------------------------------------------------
# 2-3) Save Model & Other StateDict
# --------------------------------------------------------------
torch.save({
    'model': model3.state_dict(),
    'optimizer': optimizer.state_dict()
}, Save_Path + 'all_state_dict3.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능
