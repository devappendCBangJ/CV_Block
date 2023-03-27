# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import torch
import torchvision
import torch.onnx

# ==============================================================
# 1. Model Load & Data Load
# ==============================================================
# 1) Model 생성
model = torchvision.models.vgg16(pretrained=False)
# 2) Model Parameter 저장 by OrderedDict 형태
params = model.state_dict()
# 3) Data 불러오기
dummy_data = torch.empty(1, 3, 224, 224, dtype = torch.float32)

# ==============================================================
# 2.Onnx File Save
# ==============================================================
torch.onnx.export(model, dummy_data, "output.onnx")
