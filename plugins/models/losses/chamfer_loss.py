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
class ChamferLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(ChamferLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, inputs, targets, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        weight = torch.mean(weight, dim=-1)
        loss = chamfer_distance_loss(
            inputs, targets, weight, avg_factor=avg_factor, reduction=reduction
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


@weighted_loss
def chamfer_distance_loss(pred, gt):
    """
    Calculates Chamfer Distance loss between predictions and ground truth.

    Args:
    - pred: Predicted points, tensor of shape (B, N, 3), where B is batch size, N is number of points.
    - gt: Ground truth points, tensor of the same shape as pred.

    Returns:
    - loss: Chamfer Distance loss.
    """
    B, N, _ = pred.shape

    # Expand dims to (B, N, 1, 3) for pred and gt for broadcasting
    pred_expanded = pred.unsqueeze(2)
    gt_expanded = gt.unsqueeze(1)

    # Compute squared distances (B, N, N)
    dist_squared = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)

    # For each point in pred, find min squared distance in gt
    min_dist_pred_to_gt, _ = torch.min(dist_squared, dim=2)
    # loss_pred_to_gt = torch.mean(min_dist_pred_to_gt)

    # For each point in gt, find min squared distance in pred
    min_dist_gt_to_pred, _ = torch.min(dist_squared, dim=1)
    # loss_gt_to_pred = torch.mean(min_dist_gt_to_pred)

    # Total Chamfer Distance loss is the sum of both components
    loss = min_dist_gt_to_pred + min_dist_pred_to_gt
    # loss = loss_pred_to_gt + loss_gt_to_pred

    return loss
