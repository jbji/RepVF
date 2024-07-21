# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32
from mmdet.core import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
)
from mmdet.models import HEADS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid

from openlanev1.utils import *

from mmcv.cnn import (
    Conv2d,
    Linear,
    bias_init_with_prob,
    build_activation_layer,
    constant_init,
    kaiming_init,
    xavier_init,
)


from . import PETRHead_uni
from ...core.bbox.util import normalize_bbox


@HEADS.register_module()
class PETRHead_uni_v10(PETRHead_uni):
    def _init_geometry_branch(self, pred_size=5):
        super()._init_geometry_branch(pred_size)

    def init_weights(self):
        """Initialize weights of the transformer head."""

        self.transformer.init_weights()
        # nn.init.uniform_(self.reference_point_sets.weight.data, 0, 1)

        with torch.no_grad():
            for i in range(self.num_queries_sets):
                # Create a random point in 3D space
                random_point = torch.rand(3)
                # Repeat the point num_points_per_set times
                repeated_point_set = random_point.repeat(
                    self.num_points_per_set, 1
                ).view(-1)
                # Assign this set to the i-th embedding entry
                self.reference_point_sets.weight.data[i] = repeated_point_set

        if self.loss_bbox_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.semantic_branch:
                nn.init.constant_(m[-1].bias, bias_init)

    def _init_interpreters(self):
        self.bbox3d_heading_calc = VectorField2DHeading()
        self.bbox3d_geometry_interpreter = DirectionVectorStdBBoxInterpreter()
        self.lane3d_geometry_interpreter = PointSortInterpreter()

    def _parameter_free_transduction(self, all_outs_geometry, all_outs_semantic):
        bbox3d_geometry = all_outs_geometry[:, :, : self.num_queries_sets_task[0], :]
        bbox3d_semantic = all_outs_semantic[:, :, : self.num_queries_sets_task[0], :]
        bbox3d_heading = self.bbox3d_heading_calc(bbox3d_geometry)
        bbox3d_geometry = self.bbox3d_geometry_interpreter(
            bbox3d_geometry[..., :3], bbox3d_heading
        )
        bbox3d_semantic = group_semantics(bbox3d_semantic, self.bbox_cls_out_channels)

        lane3d_geometry = all_outs_geometry[
            :, :, self.num_queries_sets_task[0] : self.num_queries_sets, :
        ]
        lane3d_semantic = all_outs_semantic[
            :, :, self.num_queries_sets_task[0] : self.num_queries_sets, :
        ]
        lane3d_geometry = self.lane3d_geometry_interpreter(
            lane3d_geometry, field_dims=5
        )
        # lane3d_geometry = lane3d_geometry.reshape(*lane3d_geometry.shape[:-2], -1)
        lane3d_semantic = group_semantics(lane3d_semantic, self.lane3d_cls_out_channels)

        return bbox3d_geometry, bbox3d_semantic, lane3d_geometry, lane3d_semantic

    def loss_single_lane3d(
        self,
        preds_xyz,
        cls_scores,
        gt_xyz_list,
        gt_cls_list,
        # vis_scores, gt_vis_list
    ):
        batch_size = preds_xyz.size(0)
        preds_xyz_list = [preds_xyz[i][..., :3] for i in range(batch_size)]
        cls_scores_list = [cls_scores[i] for i in range(batch_size)]
        # vis_scores_list = [vis_scores[i] for i in range(batch_size)]
        cls_reg_vis_targets = self.get_targets_lane3d(
            preds_xyz_list,
            cls_scores_list,
            # vis_scores_list,
            gt_xyz_list,
            gt_cls_list,
            # gt_vis_list,
        )
        (
            labels_list,
            label_weights_list,
            lane_targets_list,
            lane_weights_list,
            # vis_list,
            # vis_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_vis_targets

        # Concatenate tensors from all images in the batch
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        lane_targets = torch.cat(lane_targets_list, 0)
        lane_weights = torch.cat(lane_weights_list, 0)
        # vis_targets = torch.cat(vis_list, 0)
        # vis_weights = torch.cat(vis_weights_list, 0)

        # [bs, num_query, num_category], [200]
        cls_scores = cls_scores.reshape(-1, self.lane3d_params_dict["num_lane_cls"])
        # this avg factor calculating process seems a dummy one.
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_lane3d_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        preds_xyz = preds_xyz.reshape(-1, self.num_points_per_set, 5)
        # calculate ground truth lane field here
        # lane_targets.shape = [n,20,3] -> [n,20,5]
        # target vector
        t_vec = torch.zeros_like(lane_targets)
        t_vec[..., :-1, :] = lane_targets[..., 1:, :] - lane_targets[..., :-1, :]
        t_vec[..., -1, :] = t_vec[..., -2, :]
        t_vec[..., 2] = 0
        t_vec = F.normalize(t_vec, dim=-1, p=2)
        lane_targets = torch.cat(
            [lane_targets, t_vec[..., 1:2], t_vec[..., 0:1]], dim=-1
        )  # sine first then cosine
        # assume xyz share same weight, so is sine and cosine
        lane_weights = torch.cat(
            [lane_weights, lane_weights[..., 2:3], lane_weights[..., 2:3]], dim=-1
        )
        loss_xyz = self.loss_lane3d_reg(
            preds_xyz, lane_targets, lane_weights, avg_factor=num_total_pos
        )

        # Visibility prediction loss (replace with your actual loss function)
        # vis_scores = vis_scores.reshape(-1, self.num_points_per_set)
        # vis_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        # vis_avg_factor = max(vis_avg_factor, 1)
        # loss_vis = self.loss_lane3d_vis(
        #     vis_scores, vis_targets, vis_weights, avg_factor=vis_avg_factor
        # )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_xyz = torch.nan_to_num(loss_xyz)
        # loss_vis = torch.nan_to_num(loss_vis)
        # Return the calculated losses
        return loss_cls, loss_xyz  # , loss_vis
        # return self.loss_lane3d_anchor(preds_3dlane, gt_3dlane_anchor)

    def loss_single_bbox3d(
        self,
        cls_scores,
        bbox_preds,
        point_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.bbox_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_bbox_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        # use gt heading to recalculate the bbox.
        point_preds = point_preds.reshape(-1, point_preds.size(-2), 5)
        bbox_preds_new = self.bbox3d_geometry_interpreter(
            point_preds[..., :3], normalized_bbox_targets[..., -2:]
        )
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        bbox_preds_new = bbox_preds_new.reshape(-1, bbox_preds_new.size(-1))
        bbox_preds_to_use = torch.concatenate(
            [bbox_preds_new[..., :-2], bbox_preds[..., -2:]], dim=-1
        )

        loss_bbox = self.loss_bbox(
            bbox_preds_to_use[isnotnan, : self.code_size],
            normalized_bbox_targets[isnotnan, : self.code_size],
            bbox_weights[isnotnan, : self.code_size],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_bbox3d(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        # just adds points input for loss calc.

        loss_dict = dict()
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        # empty ground truth
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        all_bbox_points = preds_dicts["all_outs_geometry"][
            :, :, : self.num_queries_sets_task[0]
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single_bbox3d,
            all_cls_scores,
            all_bbox_preds,
            all_bbox_points,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_bboxes_ignore_list,
        )

        # loss from other decoder layers
        num_dec_layer, w = 0, 0.3
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = w * loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = w * loss_bbox_i
            num_dec_layer += 1
            w = (num_dec_layer + 1) * 0.3

        # loss from the last decoder layer
        loss_dict["loss_cls"] = w * losses_cls[-1]
        loss_dict["loss_bbox"] = w * losses_bbox[-1]

        return loss_dict


@HEADS.register_module()
class PETRHead_uni_v10_1(PETRHead_uni_v10):
    """Use seperate loss for angle prediction.

    Args:
        PETRHead_uni_v10 (_type_): _description_
    """

    def _before_init_layers(self):
        self.loss_bbox_ang = build_loss(self.bbox3d_params_dict["loss_ang"])
        self.loss_lane3d_ang = build_loss(self.lane3d_params_dict["loss_ang"])
        super()._before_init_layers()

    def loss_single_lane3d(
        self,
        preds_xyz,
        cls_scores,
        gt_xyz_list,
        gt_cls_list,
        # vis_scores, gt_vis_list
    ):
        batch_size = preds_xyz.size(0)
        preds_xyz_list = [preds_xyz[i][..., :3] for i in range(batch_size)]
        cls_scores_list = [cls_scores[i] for i in range(batch_size)]
        # vis_scores_list = [vis_scores[i] for i in range(batch_size)]
        cls_reg_vis_targets = self.get_targets_lane3d(
            preds_xyz_list,
            cls_scores_list,
            # vis_scores_list,
            gt_xyz_list,
            gt_cls_list,
            # gt_vis_list,
        )
        (
            labels_list,
            label_weights_list,
            lane_targets_list,
            lane_weights_list,
            # vis_list,
            # vis_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_vis_targets

        # Concatenate tensors from all images in the batch
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        lane_targets = torch.cat(lane_targets_list, 0)
        lane_weights = torch.cat(lane_weights_list, 0)
        # vis_targets = torch.cat(vis_list, 0)
        # vis_weights = torch.cat(vis_weights_list, 0)

        # [bs, num_query, num_category], [200]
        cls_scores = cls_scores.reshape(-1, self.lane3d_params_dict["num_lane_cls"])
        # this avg factor calculating process seems a dummy one.
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_lane3d_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        preds_xyz = preds_xyz.reshape(-1, self.num_points_per_set, 5)
        # calculate ground truth lane field here
        # lane_targets.shape = [n,20,3] -> [n,20,5]
        # target vector
        t_vec = torch.zeros_like(lane_targets)
        t_vec[..., :-1, :] = lane_targets[..., 1:, :] - lane_targets[..., :-1, :]
        t_vec[..., -1, :] = t_vec[..., -2, :]
        t_vec[..., 2] = 0
        t_vec = F.normalize(t_vec, dim=-1, p=2)
        lane_targets = torch.cat(
            [lane_targets, t_vec[..., 1:2], t_vec[..., 0:1]], dim=-1
        )  # sine first then cosine
        # assume xyz share same weight, so is sine and cosine
        lane_weights = torch.cat(
            [lane_weights, lane_weights[..., 2:3], lane_weights[..., 2:3]], dim=-1
        )
        loss_xyz = self.loss_lane3d_reg(
            preds_xyz[..., :3],
            lane_targets[..., :3],
            lane_weights[..., :3],
            avg_factor=num_total_pos,
        )
        loss_ang = self.loss_lane3d_ang(
            preds_xyz[..., 3:],
            lane_targets[..., 3:],
            lane_weights[..., 3:],
            avg_factor=num_total_pos,
        )

        # Visibility prediction loss (replace with your actual loss function)
        # vis_scores = vis_scores.reshape(-1, self.num_points_per_set)
        # vis_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        # vis_avg_factor = max(vis_avg_factor, 1)
        # loss_vis = self.loss_lane3d_vis(
        #     vis_scores, vis_targets, vis_weights, avg_factor=vis_avg_factor
        # )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_xyz = torch.nan_to_num(loss_xyz)
        loss_ang = torch.nan_to_num(loss_ang)
        # loss_vis = torch.nan_to_num(loss_vis)
        # Return the calculated losses
        return loss_cls, loss_xyz, loss_ang  # , loss_vis
        # return self.loss_lane3d_anchor(preds_3dlane, gt_3dlane_anchor)

    def loss_single_bbox3d(
        self,
        cls_scores,
        bbox_preds,
        point_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.bbox_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_bbox_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        # use gt heading to recalculate the bbox.
        bbox_preds_to_use = self.bbox_gt_recalc(
            bbox_preds, point_preds, normalized_bbox_targets
        )

        loss_bbox = self.loss_bbox(
            bbox_preds_to_use[isnotnan, : self.code_size - 2],
            normalized_bbox_targets[isnotnan, : self.code_size - 2],
            bbox_weights[isnotnan, : self.code_size - 2],
            avg_factor=num_total_pos,
        )

        loss_ang = self.loss_bbox_ang(
            bbox_preds_to_use[isnotnan, self.code_size - 2 :],
            normalized_bbox_targets[isnotnan, self.code_size - 2 :],
            bbox_weights[isnotnan, self.code_size - 2 :],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_ang = torch.nan_to_num(loss_ang)
        return loss_cls, loss_bbox, loss_ang

    def bbox_gt_recalc(
        self, bbox_preds, point_preds, normalized_bbox_targets, pred_size=5
    ):
        point_preds = point_preds.reshape(-1, point_preds.size(-2), pred_size)
        bbox_preds_new = self.bbox3d_geometry_interpreter(
            point_preds[..., :3], normalized_bbox_targets[..., -2:]
        )
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        bbox_preds_new = bbox_preds_new.reshape(-1, bbox_preds_new.size(-1))
        bbox_preds_to_use = torch.concatenate(
            [bbox_preds_new[..., :-2], bbox_preds[..., -2:]], dim=-1
        )

        return bbox_preds_to_use

    def loss_bbox3d(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        # just adds points input for loss calc.

        loss_dict = dict()
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        # empty ground truth
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        all_bbox_points = preds_dicts["all_outs_geometry"][
            :, :, : self.num_queries_sets_task[0]
        ]

        losses_cls, losses_bbox, losses_ang = multi_apply(
            self.loss_single_bbox3d,
            all_cls_scores,
            all_bbox_preds,
            all_bbox_points,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_bboxes_ignore_list,
        )

        # loss from other decoder layers
        num_dec_layer, w = 0, 0.3
        for loss_cls_i, loss_bbox_i, loss_ang_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_ang[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = w * loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = w * loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_ang"] = w * loss_ang_i
            num_dec_layer += 1
            w = (num_dec_layer + 1) * 0.3

        # loss from the last decoder layer
        loss_dict["loss_cls"] = w * losses_cls[-1]
        loss_dict["loss_bbox"] = w * losses_bbox[-1]
        loss_dict["loss_ang"] = w * losses_ang[-1]

        return loss_dict

    def loss_lane3d(
        self,
        gt_lanes_3d,
        preds_dicts,
    ):
        loss_dict = dict()
        all_lane3d_preds_xyz = preds_dicts["all_lane3d_preds_xyz"]
        all_lane3d_cls_scores = preds_dicts["all_lane3d_cls_scores"]
        # all_lane3d_vis_scores = preds_dicts["all_lane3d_vis_scores"]

        num_dec_layers = len(all_lane3d_preds_xyz)
        all_lane3d_gt_xyz_list = [
            gt_lanes_3d["lane_pts"] for _ in range(num_dec_layers)
        ]
        all_lane3d_gt_cls_list = [
            gt_lanes_3d["lane_category"] for _ in range(num_dec_layers)
        ]
        # all_lane3d_gt_vis_list = [
        # gt_lanes_3d["lane_visibility"] for _ in range(num_dec_layers)
        # ]

        # multi apply means apply the function to each element in the list
        losses_lane3d_cls, losses_lane3d_xyz, losses_lane3d_ang = multi_apply(
            self.loss_single_lane3d,
            all_lane3d_preds_xyz,
            all_lane3d_cls_scores,
            # all_lane3d_vis_scores,
            all_lane3d_gt_xyz_list,
            all_lane3d_gt_cls_list,
            # all_lane3d_gt_vis_list,
        )

        # loss from other decoder layers
        num_dec_layer, w = 0, 0.3
        for (
            loss_lane3d_cls_i,
            # loss_lane3d_vis_i,
            loss_lane3d_xyz_i,
            loss_lane3d_ang_i,
        ) in zip(
            losses_lane3d_cls[:-1],
            # losses_lane3d_vis[:-1],
            losses_lane3d_xyz[:-1],
            losses_lane3d_ang[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_lane3d_cls"] = loss_lane3d_cls_i * w
            # loss_dict[f"d{num_dec_layer}.loss_lane3d_vis"] = loss_lane3d_vis_i
            loss_dict[f"d{num_dec_layer}.loss_lane3d_xyz"] = loss_lane3d_xyz_i * w
            loss_dict[f"d{num_dec_layer}.loss_lane3d_ang"] = loss_lane3d_ang_i * w
            num_dec_layer += 1
            w = (num_dec_layer + 1) * 0.3

        # loss from the last decoder layer
        loss_dict["loss_lane3d_cls"] = losses_lane3d_cls[-1] * w
        # loss_dict["loss_lane3d_vis"] = losses_lane3d_vis[-1]
        loss_dict["loss_lane3d_xyz"] = losses_lane3d_xyz[-1] * w
        loss_dict["loss_lane3d_ang"] = losses_lane3d_ang[-1] * w
        return loss_dict


@HEADS.register_module()
class PETRHead_uni_v10_1_nus(PETRHead_uni_v10_1):
    def _init_geometry_branch(self, pred_size=7):
        super()._init_geometry_branch(pred_size)

    def _parameter_free_transduction(self, all_outs_geometry, all_outs_semantic):
        bbox3d_geometry = all_outs_geometry[:, :, : self.num_queries_sets_task[0], :]
        bbox3d_semantic = all_outs_semantic[:, :, : self.num_queries_sets_task[0], :]
        bbox3d_heading = self.bbox3d_heading_calc(bbox3d_geometry)
        bbox3d_velocity = bbox3d_geometry[..., -2:].mean(dim=-2)
        bbox3d_geometry = self.bbox3d_geometry_interpreter(
            bbox3d_geometry[..., :3], bbox3d_heading
        )
        bbox3d_geometry = torch.cat([bbox3d_geometry, bbox3d_velocity], dim=-1)
        bbox3d_semantic = group_semantics(bbox3d_semantic, self.bbox_cls_out_channels)

        lane3d_geometry = all_outs_geometry[
            :, :, self.num_queries_sets_task[0] : self.num_queries_sets, :
        ]
        lane3d_semantic = all_outs_semantic[
            :, :, self.num_queries_sets_task[0] : self.num_queries_sets, :
        ]
        lane3d_geometry = self.lane3d_geometry_interpreter(
            lane3d_geometry, field_dims=5
        )
        # lane3d_geometry = lane3d_geometry.reshape(*lane3d_geometry.shape[:-2], -1)
        lane3d_semantic = group_semantics(lane3d_semantic, self.lane3d_cls_out_channels)

        return bbox3d_geometry, bbox3d_semantic, lane3d_geometry, lane3d_semantic

    def bbox_gt_recalc(
        self, bbox_preds, point_preds, normalized_bbox_targets, pred_size=7
    ):
        point_preds = point_preds.reshape(-1, point_preds.size(-2), pred_size)
        bbox_preds_new = self.bbox3d_geometry_interpreter(
            point_preds[..., :3], normalized_bbox_targets[..., -4:-2]
        )
        bbox_velocity = point_preds[..., -2:].mean(-2)
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        bbox_velocity = bbox_velocity.reshape(-1, bbox_velocity.size(-1))
        bbox_preds_new = bbox_preds_new.reshape(-1, bbox_preds_new.size(-1))
        bbox_preds_to_use = torch.concatenate(
            [bbox_preds_new[..., :-2], bbox_preds[..., -4:-2], bbox_velocity], dim=-1
        )

        return bbox_preds_to_use


@HEADS.register_module()
class PETRHead_uni_v10_1_nus_angle_fix(PETRHead_uni_v10_1_nus):
    def loss_single_bbox3d(
        self,
        cls_scores,
        bbox_preds,
        point_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.bbox_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_bbox_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        # use gt heading to recalculate the bbox.
        bbox_preds_to_use = self.bbox_gt_recalc(
            bbox_preds, point_preds, normalized_bbox_targets
        )

        # loss_bbox = self.loss_bbox(
        #     bbox_preds_to_use[isnotnan, : self.code_size - 2],
        #     normalized_bbox_targets[isnotnan, : self.code_size - 2],
        #     bbox_weights[isnotnan, : self.code_size - 2],
        #     avg_factor=num_total_pos,
        # )

        # loss_ang = self.loss_bbox_ang(
        #     bbox_preds_to_use[isnotnan, self.code_size - 2 :],
        #     normalized_bbox_targets[isnotnan, self.code_size - 2 :],
        #     bbox_weights[isnotnan, self.code_size - 2 :],
        #     avg_factor=num_total_pos,
        # )

        # consider use code_weights in the future
        loss_bbox = self.loss_bbox(
            torch.cat(
                [
                    bbox_preds_to_use[isnotnan, : self.code_size - 4],
                    bbox_preds_to_use[isnotnan, self.code_size - 2 :],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    normalized_bbox_targets[isnotnan, : self.code_size - 4],
                    normalized_bbox_targets[isnotnan, self.code_size - 2 :],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    bbox_weights[isnotnan, : self.code_size - 4],
                    bbox_weights[isnotnan, self.code_size - 2 :],
                ],
                dim=1,
            ),
            avg_factor=num_total_pos,
        )

        loss_ang = self.loss_bbox_ang(
            bbox_preds_to_use[isnotnan, self.code_size - 4 : self.code_size - 2],
            normalized_bbox_targets[isnotnan, self.code_size - 4 : self.code_size - 2],
            bbox_weights[isnotnan, self.code_size - 4 : self.code_size - 2],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_ang = torch.nan_to_num(loss_ang)
        return loss_cls, loss_bbox, loss_ang


@HEADS.register_module()
class PETRHead_uni_v10_1_headingfix(PETRHead_uni_v10_1):
    def _init_interpreters(self):
        self.bbox3d_heading_calc = VectorField2DHeading()
        self.bbox3d_geometry_interpreter = DirectionVectorStdBBoxInterpreterHeadingFix()
        self.lane3d_geometry_interpreter = PointSortInterpreter()


@HEADS.register_module()
class PETRHead_uni_v10_2(PETRHead_uni_v10_1):
    def _init_interpreters(self):
        self.bbox3d_heading_calc = VectorField2DHeading()
        self.bbox3d_geometry_interpreter = DirectionVectorMinMaxBBoxInterpreter()
        self.lane3d_geometry_interpreter = PointSortInterpreter()


@HEADS.register_module()
class PETRHead_uni_v10_1_laneonly(PETRHead_uni_v10_1):

    def _parameter_free_transduction(self, all_outs_geometry, all_outs_semantic):
        # bbox3d_geometry = all_outs_geometry[:, :, : self.num_queries_sets_task[0], :]
        # bbox3d_semantic = all_outs_semantic[:, :, : self.num_queries_sets_task[0], :]
        # bbox3d_geometry = self.bbox3d_geometry_interpreter(bbox3d_geometry)
        # bbox3d_semantic = group_semantics(bbox3d_semantic, self.bbox_cls_out_channels)

        bbox3d_geometry, bbox3d_semantic = None, None

        lane3d_geometry = all_outs_geometry[
            :, :, self.num_queries_sets_task[0] : self.num_queries_sets, :
        ]
        lane3d_semantic = all_outs_semantic[
            :, :, self.num_queries_sets_task[0] : self.num_queries_sets, :
        ]
        # lane3d_geometry = lane3d_geometry.reshape(*lane3d_geometry.shape[:-2], -1)
        lane3d_semantic = group_semantics(lane3d_semantic, self.lane3d_cls_out_channels)

        return bbox3d_geometry, bbox3d_semantic, lane3d_geometry, lane3d_semantic

    def _init_interpreters(self):
        self.bbox3d_heading_calc = None
        self.bbox3d_geometry_interpreter = None
        self.lane3d_geometry_interpreter = PointSortInterpreter()

    def forward(self, mlvl_feats, img_metas):
        outs = super().forward(mlvl_feats, img_metas)
        outs["all_cls_scores"] = None
        outs["all_bbox_preds"] = None
        return outs

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_lanes_3d,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        # return dict
        loss_dict = dict()

        # 2 3d object detection loss, skip for now
        # if self.bbox3d_enabled:
        # loss_dict_bbox3d = self.loss_bbox3d(
        # gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore
        # )
        # loss_dict.update(loss_dict_bbox3d)

        if self.lane3d_enabled:
            loss_dict_lane3d = self.loss_lane3d(gt_lanes_3d, preds_dicts)
            loss_dict.update(loss_dict_lane3d)

        return loss_dict


@HEADS.register_module()
class PETRHead_uni_v10_1_detonly(PETRHead_uni_v10_1):
    def forward(self, mlvl_feats, img_metas):
        outs = super().forward(mlvl_feats, img_metas)
        outs["all_lane3d_preds_xyz"] = None
        outs["all_lane3d_cls_scores"] = None
        return outs

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_lanes_3d,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        # return dict
        loss_dict = dict()

        # 2 3d object detection loss, skip for now
        if self.bbox3d_enabled:
            loss_dict_bbox3d = self.loss_bbox3d(
                gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore
            )
            loss_dict.update(loss_dict_bbox3d)

        # if self.lane3d_enabled:
        # loss_dict_lane3d = self.loss_lane3d(gt_lanes_3d, preds_dicts)
        # loss_dict.update(loss_dict_lane3d)

        return loss_dict


@HEADS.register_module()
class PETRHead_uni_v10_1_detonly_nus(PETRHead_uni_v10_1_nus):
    def forward(self, mlvl_feats, img_metas):
        outs = super().forward(mlvl_feats, img_metas)
        outs["all_lane3d_preds_xyz"] = None
        outs["all_lane3d_cls_scores"] = None
        return outs

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_lanes_3d,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        # return dict
        loss_dict = dict()

        # 2 3d object detection loss, skip for now
        if self.bbox3d_enabled:
            loss_dict_bbox3d = self.loss_bbox3d(
                gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore
            )
            loss_dict.update(loss_dict_bbox3d)

        # if self.lane3d_enabled:
        # loss_dict_lane3d = self.loss_lane3d(gt_lanes_3d, preds_dicts)
        # loss_dict.update(loss_dict_lane3d)

        return loss_dict


@HEADS.register_module()
class PETRHead_uni_v10_1_detonly_nus_angle_fix(PETRHead_uni_v10_1_nus_angle_fix):
    def forward(self, mlvl_feats, img_metas):
        outs = super().forward(mlvl_feats, img_metas)
        outs["all_lane3d_preds_xyz"] = None
        outs["all_lane3d_cls_scores"] = None
        return outs

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_lanes_3d,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        # return dict
        loss_dict = dict()

        # 2 3d object detection loss, skip for now
        if self.bbox3d_enabled:
            loss_dict_bbox3d = self.loss_bbox3d(
                gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore
            )
            loss_dict.update(loss_dict_bbox3d)

        # if self.lane3d_enabled:
        # loss_dict_lane3d = self.loss_lane3d(gt_lanes_3d, preds_dicts)
        # loss_dict.update(loss_dict_lane3d)

        return loss_dict
