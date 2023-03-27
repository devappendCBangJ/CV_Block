# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
import os
from PIL import Image
import requests

# ==============================================================
# 1. url 이미지 저장
# ==============================================================
# 1) url 저장
image_url = ('https://ee5817f8e2e9a2e34042-3365e7f0719651e5b'
             '8d0979bce83c558.ssl.cf5.rackcdn.com/python.png')
# 2) url 내용 추출
image = requests.get(image_url).content  # 서버 응답을 받아 파일내용 획득. content는 응답받은 RawData
# 3) url 파일명 -> 깡통 저장
filename = os.path.basename(image_url)  # URL에서 파일명 획득. 뒷부분의 python.png란 파일명만 저장
# 4) url 내용 -> 파일 저장
with open(filename, 'wb') as f:  # wb: 쓰기 바이너리
    f.write(image)  # 파일 저장

# ==============================================================
# - 이미지 출력
# ==============================================================
pil_img = Image.open(filename)
pil_img.show()

# ==============================================================
# 2. 이미지 해상도 압축
# ==============================================================
filename_40 = filename[:-4] + "_40" + ".jpg"
with Image.open(filename) as im:
    im = im.convert("RGB")
    im.save(filename_40, quality=40) #quality는 jpg포맷만 유효

# ==============================================================
# - 이미지 출력
# ==============================================================
pil_img = Image.open(filename_40)
pil_img.show()

# ==============================================================
# 3. 이미지 해상도 압축 -> 사이즈 압축
# ==============================================================
filename_40_thumbnail = filename[:-4] + "_40_thumbnail" + ".jpg"
with Image.open(filename_40) as im:
    im.thumbnail((300, 300))#원본을 300 by 300 변경
    im.save(filename_40_thumbnail) #quality는 jpg포맷만 유효

# ==============================================================
# - 이미지 출력
# ==============================================================
pil_img = Image.open(filename_40_thumbnail)
pil_img.show()

# ==============================================================
# 4. 이미지 사이즈 압축
# ==============================================================
filename_thumbnail = filename[:-4] + "_thumbnail" + filename[-4:]
with Image.open(filename) as im:
    im.thumbnail((300, 300))#원본을 300 by 300 변경
    im.save(filename_thumbnail) #quality는 jpg포맷만 유효

# ==============================================================
# - 이미지 출력
# ==============================================================
pil_img = Image.open(filename_thumbnail)
pil_img.show()

# ==============================================================
# 5. 이미지 해상도 압축 + 사이즈 압축
# ==============================================================
filename_40_thumbnail_all = filename[:-4] + "_40_thumbnail_all" + ".jpg"
with Image.open(filename) as im:
    im = im.convert("RGB")
    im.thumbnail((300, 300))  # 원본을 300 by 300 변경
    im.save(filename_40_thumbnail_all, quality=40)  # quality는 jpg포맷만 유효

# ==============================================================
# - 이미지 출력
# ==============================================================
pil_img = Image.open(filename_40_thumbnail_all)
pil_img.show()
