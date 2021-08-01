import torch.nn as nn
import torch.nn.functional as F

from resnet import get_backbone
from neck import get_neck
from rpn import get_rpn_head


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(**{'used_layers': [2, 3, 4]})

        # build adjust layer
        self.neck = get_neck(
            **{'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]})

        # build rpn head
        self.rpn_head = get_rpn_head(
            **{'anchor_num': 5, 'in_channels': [256, 256, 256], 'weighted': True})

    def template(self, z):
        zf = self.backbone(z)
        zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)

        return {
            'cls': cls,
            'loc': loc,
        }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.neck(zf)
        xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return outputs
