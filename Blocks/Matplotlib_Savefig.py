# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. X, Y 생성
# ==============================================================
# 1) X
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

# 2) y
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

# ==============================================================
# 2. Visualization
# ==============================================================
# 1) Plt Subplot Split
plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1
# 2) Plt Values
plt.plot(x1, y1, 'o-')
# 3) Plt Title
plt.title('1st Graph')
# 4) Plt Label
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
plt.plot(x2, y2, '.-')
plt.title('2nd Graph')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.tight_layout()
# plt.show()

# ==============================================================
# 2. Save Image from Plot
# ==============================================================
# 저장 경로 / dpi : 해상도 / facecolor : 이미지의 배경색 / edgecolor : 테두리 색 / bbox_inches : 저장할 이미지 영역 / pad_inches : bbox_inches와 함게 여백 너비
plt.savefig('savefig_default.png', dpi=200, facecolor='#eeeeee', edgecolor='black', bbox_inches='tight', pad_inches=0.2)