from get_frame import get_frames
from IOU import IOU
import os
import argparse

import cv2
import torch

from model import ModelBuilder
from tracker import build_tracker

parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--type', default='2D', type=str,
                    help='2D video or 3D video')
parser.add_argument('--img2d_ref', default='images/005.png',
                    type=str, help='Main image root')
parser.add_argument('--gt_on', default=False, type=bool, help='Estimate IoU')
parser.add_argument('--record', default=False, type=bool,
                    help='Save images and IoU accuracy')
parser.add_argument('--start_num', default=20, type=int,
                    help='First focal image number')
parser.add_argument('--last_num', default=50, type=int,
                    help='Last focal image number')
args = parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    # ground truth
    gt_on = args.gt_on  # IoU 정확도를 측정할 것인지
    f = open('ground_truth/Non_video4_GT.txt', 'r')  # GT 파일

    # create model
    model = ModelBuilder()

    # load model
    checkpoint = torch.load("pretrained_model/model.pth",
                            map_location=lambda storage, loc: storage.cpu())

    model.load_state_dict(checkpoint)
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    video_name = args.video_name.split('/')[-1].split('.')[0]
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    frame_num = 0
    for frame in get_frames(args.video_name, args.type, args.img2d_ref, args.start_num, args.last_num):
        frame_num += 1
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))

            #### ground truth ####
            if gt_on:
                line = f.readline()
                bbox_label = line.split(',')
                bbox_label = list(map(int, bbox_label))

                labelx = bbox_label[0] + (bbox_label[2] / 2)
                labely = bbox_label[1] + (bbox_label[3] / 2)

                iou = IOU(bbox, bbox_label)
                pre = ((outputs['cx'] - labelx)**2 +
                       (outputs['cy'] - labely)**2) ** 0.5

                if args.record:
                    result_iou = open('ground_truth/result_iou.txt', 'a')
                    result_iou.write(str(iou) + ',')
                    result_iou.close()

                    result_pre = open('ground_truth/result_pre.txt', 'a')
                    result_pre.write(str(pre) + ',')
                    result_pre.close()

                cv2.rectangle(frame, (bbox_label[0], bbox_label[1]),
                              (bbox_label[0]+bbox_label[2],
                               bbox_label[1]+bbox_label[3]),
                              (255, 255, 255), 3)

            #### ----------------- ####

            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 0, 255), 3)
            cv2.imshow(video_name, frame)
            if args.record:
                save_image(frame_num, frame)
            cv2.waitKey(40)


def save_image(frame_num, frame):
    '''output 이미지 저장'''
    save_path = os.path.join(
        'data/result', '{:03d}.jpg'.format(frame_num))
    cv2.imwrite(save_path, frame)
    ''''''


if __name__ == "__main__":
    main()
