import os
import cv2
import torch
import time
from pathlib import Path, PurePosixPath
from datetime import datetime

from config_parser import ConfigParser
from get_frame import get_frames
from IOU import IOU
from model import ModelBuilder
from tracker import build_tracker
from copy import deepcopy

import sys
import numpy as np


def save_image(save_dir, frame_num, frame):
    ''' output 이미지 저장 '''
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / '{:03d}.jpg'.format(frame_num)
    cv2.imwrite(str(save_path), frame)


def ground_truth(center, size):
    f = open("ground_truth/Video3.txt", 'a')
    data = "%d,%d,%d,%d\n" % (center[0], center[1], size[0], size[1])
    f.write(data)
    f.close()

def main():
    # config parsing & setting
    config = ConfigParser('./config.json')
    exper_name = config['name']
    is_gt_on = config['is_gt_on']
    is_record = config['is_record']
    video_name = config['video_name']
    video_type = config['video_type']
    img2d_ref = config['image2d_ref']
    start_focal_num = config['start_focal_num']
    last_focal_num = config['last_focal_num']
    ckpt_path = config['pretrained_model']
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    save_path = Path(config['save_path']) / timestamp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    score_threshold = 0.3
    range_val = 30

    # ground truth
    if is_gt_on:    # IoU 정확도를 측정할 것인지
        f = open('ground_truth/Non_video4_GT.txt', 'r')  # GT 파일

    # create model
    model = ModelBuilder()

    # load model
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(ckpt)
    model.eval().to(device)

    print("Please type the number of trackers: ")
    tracker_num = int(sys.stdin.readline())

    # build tracker
    tracker = []
    for _ in range(tracker_num):
        m = deepcopy(model)
        tracker.append(build_tracker(m))
        # tracker.append(build_tracker(model))

    # tracker = build_tracker(model)
    start_time = time.time()
    prev_seconds_per_frame: float = None
    each_track_times = []


    # tracking
    is_first_frame = True
    frame_num = 0
    first_time = True
    current_target = [-1 for _ in range(tracker_num)]
    bbox = [-1 for _ in range(tracker_num)]
    color = [-1 for _ in range(tracker_num)]
    outputs = [[] for _ in range(tracker_num)]

    track_infos = []
    '''
    d = {
        'prev': previous coord
        'current': current coordinate
        'moves'
    }
    '''

    # current_target2 = -1
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame, focals in get_frames(video_name, video_type, img2d_ref, start_focal_num, last_focal_num):
        frame_num += 1
        if is_first_frame:
            try:
                for k in range(tracker_num):
                    init_rect = cv2.selectROI(video_name, frame, True, False)
                    tracker[k].init(frame, init_rect)
                    color[k] = list(np.random.random(size=3) * 256)
            except:
                exit()
            # tracker.init(frame, init_rect)
            is_first_frame = False
        else:
            max_index = [-1 for _ in range(tracker_num)]
            max_val = [0 for _ in range(tracker_num)]

            if first_time:
                for k in range(tracker_num):
                    time_start = time.time()
                    outputs[k] = [tracker[k].track(cv2.imread(f)) for f in focals]
                    for i, output in enumerate(outputs[k]):
                        if output['best_score'] >= max_val[k]:
                            max_val[k] = output['best_score']
                            max_index[k] = i

                    bbox[k] = list(map(int, outputs[k][max_index[k]]['bbox']))
                    current_target[k] = max_index[k]
                    taken_time = time.time() - time_start
                    print('tracked time for {}:'.format(k), taken_time)
                    try:
                        each_track_times[k] += taken_time
                    except:
                        each_track_times.append(taken_time)
                first_time = False
            else:
                for k in range(tracker_num):
                    time_start = time.time()
                    output_backup = outputs[k]
                    try:
                        outputs[k] = [tracker[k].track(cv2.imread(focals[i])) for i in range(
                                current_target[k] - 3, current_target[k] + 3)]
                    except IndexError as e:
                        outputs[k] = output_backup

                    # print('[[OUTPUT]]', outputs[k])
                    nothing_over = True
                    for output in outputs[k]:
                        if output['best_score'] >= score_threshold:
                            nothing_over = False
                            break

                    if nothing_over:
                        range_first = current_target[k] - range_val if current_target[k] - range_val >= 0 else 0
                        range_last = current_target[k] + range_val if current_target[k] + range_val <= len(focals) else len(focals)
                        if range_last > 60:
                            range_last = 60
                        outputs[k] = [tracker[k].track(cv2.imread(focals[i])) for i in range(range_first, range_last)]
                    else:
                        for i, output in enumerate(outputs[k]):
                            if output['best_score'] >= max_val[k]:
                                max_val[k] = output['best_score']
                                max_index[k] = i

                        bbox[k] = list(map(int, outputs[k][max_index[k]]['bbox']))

                        if max_index[k] > 3:
                            current_target[k] = current_target[k] + abs(3 - max_index[k])
                        elif max_index[k] < 3:
                            current_target[k] = current_target[k] - abs(3 - max_index[k])

                    # for i, output in enumerate(outputs[k]):
                    #     if output['best_score'] >= max_val[k]:
                    #         max_val[k] = output['best_score']
                    #         max_index[k] = i

                    bbox[k] = list(map(int, outputs[k][max_index[k]]['bbox']))

                    # if max_index[k] > 3:
                    #     current_target[k] = current_target[k] + abs(3 - max_index[k])
                    # elif max_index[k] < 3:
                    #     current_target[k] = current_target[k] - abs(3 - max_index[k])

                    taken_time = time.time() - time_start
                    print('tracked time for {}:'.format(k), taken_time)
                    try:
                        each_track_times[k] += taken_time
                    except:
                        each_track_times.append(taken_time)

            for k in range(tracker_num):
                bbox_idx = bbox[k]
                cv2.rectangle(frame, (bbox_idx[0], bbox_idx[1]),
                              (bbox_idx[0] + bbox_idx[2], bbox_idx[1] + bbox_idx[3]),
                              color[k], 3)

            # # ground_truth(outputs[max_index]['bbox'][:2],
            # #              outputs[max_index]['bbox'][2:])
            # for k in range(tracker_num):
            #     cv2.rectangle(frame, (bbox[k][0], bbox[k][1]),
            #                   (bbox[k][0]+bbox[k][2], bbox[k][1]+bbox[k][3]),
            #                   color[k], 3)


            # save_path = os.path.join(
            #     'data/result2', '{:03d}.jpg'.format(frame_num))
            # cv2.imwrite(save_path, frame)

            # # ground truth
            # if is_gt_on:
            #     line = f.readline()
            #     bbox_label = line.split(',')
            #     bbox_label = list(map(int, bbox_label))
            #
            #     iou = IOU(bbox, bbox_label)
            #
            #     labelx = bbox_label[0] + (bbox_label[2] / 2)
            #     labely = bbox_label[1] + (bbox_label[3] / 2)
            #
            #     pre = ((outputs[max_index]['cx'] - labelx)**2 +
            #            (outputs[max_index]['cy'] - labely)**2) ** 0.5
            #
            #     if is_record:
            #         result_iou = open('ground_truth/result_iou.txt', 'a')
            #         result_iou.write(str(iou) + ',')
            #         result_iou.close()
            #
            #         result_pre = open('ground_truth/result_pre.txt', 'a')
            #         result_pre.write(str(pre) + ',')
            #         result_pre.close()
            #
            #     cv2.rectangle(frame, (bbox_label[0], bbox_label[1]),
            #                   (bbox_label[0]+bbox_label[2],
            #                    bbox_label[1]+bbox_label[3]),
            #                   (255, 255, 255), 3)

            cv2.imshow(video_name, frame)

            if is_record:
                save_image(save_path, frame_num, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

            time_now = time.time()
            taken_seconds = time_now - (
                prev_seconds_per_frame if prev_seconds_per_frame is not None else start_time)  # 이 프레임에 대해 총 걸린시간: 현재 프레임이 끝난 시간- 이전 프레임 끝난 시간
            prev_seconds_per_frame = time_now  # 현재 프레임이 다시 이전 프레임이 되니까.
            print('%d seconds for frame %d' % (taken_seconds, frame_num))

            end_time = time.time()  # 전체 프로그램 측정 시간
            print('{} seconds for all'.format(end_time - start_time))

            for i, k in enumerate(each_track_times):  # 각 트래커마다 자기 자신을 찾는데 걸리는 시간의 총합
                print('%d seconds for tracker %d' % (k, i))

if __name__ == "__main__":
    main()
