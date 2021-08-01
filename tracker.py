import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

from base_tracker import SiameseTracker
from anchor import Anchors


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (255 - 127) // \
            8 + 1 + 8
        self.anchor_num = len([0.33, 0.5, 1, 2, 3]) * len([8])
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        print(self.anchors)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(8,
                          [0.33, 0.5, 1, 2, 3],
                          [8])
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(
            np.float32), yy.astype(np.float32)
        print(anchor)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(
            2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    127,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def get_cls(self, focals):

        # sharpening
        sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 9, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 9.0

        max_index = -1
        sum_cls = torch.Tensor(1, 10, 25, 25)
        for i, focal in enumerate(focals):
            focal = cv2.imread(focal)
            focal = cv2.filter2D(focal, -1, sharpening_1)
            w_z = self.size[0] + 0.5 * np.sum(self.size)
            h_z = self.size[1] + 0.5 * np.sum(self.size)
            s_z = np.sqrt(w_z * h_z)
            scale_z = 127 / s_z
            s_x = s_z * (255 / 127)
            x_crop = self.get_subwindow(focal, self.center_pos,
                                        255,
                                        round(s_x), self.channel_average)

            outputs = self.model.track(x_crop)

            if i == 0:
                sum_cls = outputs['cls']
            else:
                sum_cls = torch.cat([sum_cls, outputs['cls']], dim=1)

        # convert_score
        score = self._convert_score(sum_cls)

        ''' score값들의 평균 중 max 뽑아내는 방법 '''
        group = []
        for i in range(len(score) // 3125):
            group.append(score[3125 * i: 3125 * (i+1)])

        for i in range(len(group)):
            group[i] = np.mean(group[i])
        max_index = np.argmax(group)

        ''''''

        # best_idx = np.argmax(score)

        # max_index = best_idx // 3125

        return max_index

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, sharpening_1)
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = 127 / s_z
        s_x = s_z * (255 / 127)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    255,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        '''
        response 이미지 출력하기 위해 작성
        '''

        # res = outputs['cls'].cpu().detach().numpy()
        # # res.shape = (1, 10, 25, 25)
        # res = res[0]
        # # res.shape = (10, 25, 25)
        # # plt.imshow(img)
        # # plt.show()
        # for i in range(10):
        #     plt.subplot(2, 5, i+1)
        #     plt.imshow(res[i, :, :])
        # plt.show()

        ''''''

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * 0.04)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - 0.44) + \
            self.window * 0.44
        best_idx = np.argmax(pscore)

        ##################################################################### start
        # best_indexes = []
        # for i in pscore:
        #     if len(best_indexes) < 5:
        #         best_indexes.append(i)
        #     elif min(best_indexes) < i:
        #         best_indexes[best_indexes.index(min(best_indexes))] = i

        # bboxes = []
        # best_scores = []
        # cxs = []
        # cys = []
        # for i, bi in enumerate(best_indexes):
        #     bbox = pred_bbox[:, bi] / scale_z
        #
        #     lr = penalty[bi] * score[bi] * 0.05
        #
        #     cx = bbox[0] + self.center_pos[0]
        #     cy = bbox[1] + self.center_pos[1]
        #
        #     width = self.size[0] * (1 - lr) + bbox[2] * lr
        #     height = self.size[1] * (1 - lr) + bbox[3] * lr
        #
        #     cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
        #
        #     self.center_pos = np.array([cx, cy])
        #     self.size = np.array([width, height])
        #
        #     bbox = [cx - width / 2,
        #             cy - height / 2,
        #             width,
        #             height]
        #
        #     cxs.append(cx)
        #     cys.append(cy)
        #     bboxes.append(bbox)
        #     best_scores.append(score[bi])
        ##################################################################### end

        bbox = pred_bbox[:, best_idx] / scale_z
        # lr = penalty[best_idx] * score[best_idx] * 0.4  # cfg.TRACK.LR
        lr = penalty[best_idx] * score[best_idx] * 0.05

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        return {
            'bbox': bbox,
            'best_score': best_score,
            'cx': cx,
            'cy': cy
        }
        # return {
        #     'bbox': bboxes,
        #     'best_score': best_scores,
        #     'cx': cxs,
        #     'cy': cys
        # }


def build_tracker(model):
    return SiamRPNTracker(model)
