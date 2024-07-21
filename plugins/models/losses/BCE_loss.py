# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
import torch.nn as nn
import torch.nn.functional as F


@LOSSES.register_module()
class Lane3DVisBCELoss(nn.Module):
    def __init__(self, weight=1.0, reduction="mean", loss_weight=1.0):
        super(Lane3DVisBCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        vis_scores,
        gt_vis,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        ignore_index=-100,
        avg_non_ignore=False,
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        # should mask out the ignored elements
        valid_mask = ((gt_vis >= 0) & (gt_vis != ignore_index)).float()
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

        # average loss over non-ignored elements
        if (avg_factor is None) and avg_non_ignore and reduction == "mean":
            avg_factor = valid_mask.sum().item()
            # weighted element-wise losses
            weight = weight.float()

        # Use binary_cross_entropy_with_logits for more numerically stable output
        loss = F.binary_cross_entropy_with_logits(
            vis_scores, gt_vis.float(), reduction="none"
        )

        # Apply weight and reduction
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor
        )

        return loss * self.weight * self.loss_weight
