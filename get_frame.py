from plenoptic_dataloader import PlenopticDataLoader
import cv2
from glob import glob
import os


def get_frames(video_name, type, img2d_ref, start_num, last_num):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif type == "2D":
        dataLoader_focal = PlenopticDataLoader(
            root=video_name, img2d_ref=img2d_ref, focal_range=(start_num, last_num))
        img2d_files = dataLoader_focal.dataLoader_2d()
        # for i in range(len(img2d_files)):
        for img2d_file in img2d_files:
            frame_bgr = cv2.imread(img2d_file)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_bgr, frame_rgb
    elif type == "3D":
        dataLoader_focal = PlenopticDataLoader(
            root=video_name, img2d_ref=img2d_ref, focal_range=(start_num, last_num))
        img2d_files, focal_files = dataLoader_focal.dataLoader_focal()
        # for i in range(150):
        # for i in range(len(img2d_files)):
        # print(len(img2d_files))
        # for i in range(147, len(img2d_files)):
        for i in range(147, 147 + 257):
            frame = cv2.imread(img2d_files[i])
            yield frame, focal_files[i]
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: x.split('/')[-1].split('.')[0])
        # key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame
