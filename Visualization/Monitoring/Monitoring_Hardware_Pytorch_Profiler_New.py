# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import torch
import numpy as np
from torch import nn
import torch.profiler as profiler

# ==============================================================
# 1.Def 정의
# ==============================================================
def trace_handler(prof):
    # (1) 정렬되지 않은 모든 정보 (Name + CPU % & Time + GPU % & Time + # Call + Input Shape + Source Location) 출력
    # print(prof)

    # (2) 정렬된 일부 정보 (Name + CPU % & Time + GPU % & Time + # Call + 모든 Source Location) 출력
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=10))

    # (3) 정렬된 일부 정보 (Name + CPU % & Time + GPU % & Time + # Call) 출력
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    # (4) 정렬되지 않은 수치 정보 (Name + CPU % & Time + GPU % & Time + # Call + Input Shape) 출력
    # print(prof.key_averages())
    # print(prof.key_averages().table())

    # (5) Chrome Json 파일로 보내기
    prof.export_chrome_trace("profile_cpu_gpu.json")
    prof.export_stacks("profile_cpu_gpu.txt", "self_cuda_time_total")
    profiler.tensorboard_trace_handler("/home/hi/PycharmProjects/Test/")

# ==============================================================# ==============================================================
# {1} 최적화 없음
# ==============================================================# ==============================================================
# ==============================================================
# 2. Model 정의
# ==============================================================
class MyModule(nn.Module):
    # 1] 초기 설정
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
    # 2] 순전파
    def forward(self, input, mask):
        # [1] Profiler : Linear
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)
        # [2] Profiler : Mask Indices
        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            # print("threshold : ", threshold)
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            # print("hi_idx : ", hi_idx)
            hi_idx = torch.from_numpy(hi_idx).cuda()
            # print("hi_idx : ", hi_idx)
        return out, hi_idx

# ==============================================================
# 3.Data Load -> Model Load -> Output 추출
# ==============================================================
# 1) Input
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()
# 2) Model 선언
model = MyModule(500, 10).cuda()

# ==============================================================
# 4. CPU Profile
# ==============================================================
with profiler.profile(activities=[profiler.ProfilerActivity.CPU],
                      record_shapes = True, with_stack=True, profile_memory=True,
                      schedule = profiler.schedule(skip_first=0, wait=0, warmup=1, active=3, repeat=1),
                      on_trace_ready = trace_handler) as prof:
    for e in range(5):
        print("epoch : ", e)
        out, idx = model(input, mask)
        prof.step()

# ==============================================================# ==============================================================
# {2} 자료형 최적화 : double -> float
# ==============================================================# ==============================================================
# ==============================================================
# 2. Model 정의
# ==============================================================
class MyModule(nn.Module):
    # 1] 초기 설정
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
    # 2] 순전파
    def forward(self, input, mask):
        # [1] Profiler : Linear
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)
        # [2] Profiler : Mask Indices
        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            # print("threshold : ", threshold)
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            # print("hi_idx : ", hi_idx)
            hi_idx = torch.from_numpy(hi_idx).cuda()
            # print("hi_idx : ", hi_idx)
        return out, hi_idx

# ==============================================================
# 3.Data Load -> Model Load -> Output 추출
# ==============================================================
# 1) Input
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()
# 2) Model 선언
model = MyModule(500, 10).cuda()

# ==============================================================
# 4. CPU Profile
# ==============================================================
with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                      record_shapes = True, with_stack=True, profile_memory=True,
                      schedule = profiler.schedule(skip_first=0, wait=0, warmup=1, active=3, repeat=1),
                      on_trace_ready = trace_handler) as prof:
    for e in range(5):
        print("epoch : ", e)
        out, idx = model(input, mask)
        prof.step()

# ==============================================================# ==============================================================
# {3} 행렬 복사 최적화 : numpy를 위한 cpu 사용 -> cuda에서 gpu 사용
# ==============================================================# ==============================================================
# ==============================================================
# 2. Model 정의
# ==============================================================
class MyModule(nn.Module):
    # 1] 초기 설정
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
    # 2] 순전파
    def forward(self, input, mask):
        # [1] Profiler : Linear
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)
        # [2] Profiler : Mask Indices
        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean()
            # print("threshold : ", threshold)
            hi_idx = (mask > threshold).nonzero(as_tuple=True)
            # print("hi_idx : ", hi_idx)
        return out, hi_idx

# ==============================================================
# 3.Data Load -> Model Load -> Output 추출
# ==============================================================
# 1) Input
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()
# 2) Model 선언
model = MyModule(500, 10).cuda()

# ==============================================================
# 4. CPU Profile
# ==============================================================
with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                      record_shapes = True, with_stack=True, profile_memory=True,
                      schedule = profiler.schedule(skip_first=0, wait=0, warmup=1, active=3, repeat=1),
                      on_trace_ready = trace_handler) as prof:
    for e in range(5):
        print("epoch : ", e)
        out, idx = model(input, mask)
        prof.step()
