# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
from visdom import Visdom

import numpy as np
import torch

# ==============================================================
# 1.Visdom 객체 생성
# ==============================================================
viz = Visdom()

# ==============================================================
# 2.Visdom 데이터 출력
# ==============================================================
# 1) Text
textwindow = viz.text("Hello")

# 2) Image
image_window = viz.image(
    np.random.rand(3,256,256),
    opts=dict(
        title = "random",
        caption = "random noise"
    )
)

# 3) Images
images_window = viz.images(
    np.random.rand(10,3,64,64),
    opts=dict(
        title = "random",
        caption = "random noise"
    )
)

# 4) Plot
plot = viz.line(
    X = np.array([0, 1, 2, 3, 4]),
    Y = torch.randn(5),
)

plot = viz.line(X = np.array([5]), Y = torch.randn(1), win = plot, update = 'append')

plot = viz.line(
    X = np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y = torch.randn(10, 2),
    opts = dict(
        title = "Test",
        legend = ["1번 라인", "2번 라인"],
        showlegend = True
    )
)
