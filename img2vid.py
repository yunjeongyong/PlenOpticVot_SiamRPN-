import cv2
import os

# 0729_004759 폴더를 이용했는데, 결과에 따라 다른 폴더명을 이용하면 될 것 같애'
folder = './save/%s/' % '0801_153045'
paths = [folder + filename for filename in os.listdir(folder)]
paths.sort()

# 영상이 output.mp4라는 파일로 저장됨
output = './output.mp4'

# FPS를 3으로 설정하였는데, 변경하면 영상의 속도가 달라짐
fps = 3
frame_array = []

size = None

for i, path in enumerate(paths):
    if (i % 2 == 0) | (i % 5 == 0):
        continue
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)

out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    out.write(frame_array[i])

out.release()