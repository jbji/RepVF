import math
import warnings

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox_overlaps
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss
import numpy as np


@LOSSES.register_module()
class Line3DIoULoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(Line3DIoULoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(
        self, inputs, targets, weight=None, avg_factor=None, reduction_override=None
    ):
        loss = some_calculation(
            inputs, targets, weight, avg_factor=avg_factor, reduction=reduction_override
        )
        return self.loss_weight * loss


@weighted_loss
def some_calculation(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = None
    # some calculation
    return loss


@LOSSES.register_module()
class LineIoULoss(nn.Module):
    """From CLRNet: Cross Layer Refinement Network for Lane Detection

    Args:
        nn (_type_): _description_
    """

    def __init__(self, loss_weight=1.0):
        super(Sigmoid_ce_loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        inputs,
        targets,
    ):
        loss = (1 - line_iou(inputs, targets, img_w, length)).mean()
        return self.loss_weight * loss

    def line_iou(pred, target, img_w, length=15, aligned=True):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (num_pred, 72)
            target: ground truth, shape: (num_target, 72)
            img_w: image width
            length: extended radius
            aligned: True for iou loss calculation, False for pair-wise ious in assign
        """
        px1 = pred - length
        px2 = pred + length
        tx1 = target - length
        tx2 = target + length
        if aligned:
            invalid_mask = target
            ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
            union = torch.max(px2, tx2) - torch.min(px1, tx1)
        else:
            num_pred = pred.shape[0]
            invalid_mask = target.repeat(num_pred, 1, 1)
            ovr = torch.min(px2[:, None, :], tx2[None, ...]) - torch.max(
                px1[:, None, :], tx1[None, ...]
            )
            union = torch.max(px2[:, None, :], tx2[None, ...]) - torch.min(
                px1[:, None, :], tx1[None, ...]
            )

        invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
        ovr[invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou
