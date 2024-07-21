# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

import torch
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core import (
    CameraInstance3DBoxes,
    LiDARInstance3DBoxes,
    bbox3d2result,
    show_multi_modality_result,
)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ..utils.grid_mask import GridMask
from openlanev1.utils.npdump import npdump


@DETECTORS.register_module()
class RFTR(MVXTwoStageDetector):
    """RFTR(3D)"""

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(RFTR, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        gt_lanes_3d,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_lanes_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @force_fp32(apply_to=("img"))
    def forward_dummy(self, img):
        """Dummy forward for FLOPS calculation"""
        # should be [2,1,3,512,1408]
        # if not
        import numpy as np

        cams = 5
        single_lidar2img = np.array(
            [
                [2.07322128e03, 8.40773165e02, 1.44805734e00, -3.06390076e00],
                [1.93500047e00, 2.64287211e02, -2.07707129e03, 4.39481238e03],
                [
                    -4.01059771e-03,
                    9.99990231e-01,
                    -1.85822421e-03,
                    3.93176045e-03,
                ],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        # [single_lidar2img for i in range(5)]
        img_meta = {
            "lidar2img": [single_lidar2img for i in range(5)],
            "img_shape": [(512, 1408, 3) for i in range(5)],
            "pad_shape": [(512, 1408, 3) for i in range(5)],
        }
        img_metas = [img_meta]
        x = self.extract_feat(img=img, img_metas=img_metas)
        # should be 2 * [2,1,256,32,88]
        head_outs = self.pts_bbox_head(x, img_metas=img_metas)
        return head_outs

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        gt_lanes_3d=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_lanes_3d,
            img_metas,
            gt_bboxes_ignore,
        )
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        return self.simple_test(img_metas[0], img[0], **kwargs)

    def show_results(self, data, result, out_dir):
        super().show_results(data, result, out_dir)
        Warning("show_results is not implemented for 3D Lane Detection")

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        # how to convert lane out tensor to lane?
        bbox_results = []
        if True:
            bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]

        lane3d_results = []

        if True:
            # bs, num_lanes, lanelines_pred, lanelines_prob
            lane3d_list = self.pts_bbox_head.get_lanes(outs, img_metas, rescale=rescale)
            # lane3d_list = [[([], [])]]
            for lane_3d in lane3d_list:
                lane3d_results_dict = {}
                # take the last layer as it should be the best prediction
                lanelines_pred, lanelines_prob = lane_3d[-1]
                lane_lines = []
                # "xyz":             <float> [3, n] -- x,y,z coordinates of sample points in camera coordinate
                # "category":        <int> -- lane shape category, 1 - num_category
                for k in range(len(lanelines_pred)):
                    if np.max(lanelines_prob[k]) < 0.5:
                        continue
                    lane_lines.append(
                        {
                            "xyz": lanelines_pred[k],
                            "category": int(np.argmax(lanelines_prob[k])),
                        }
                    )
                # TODO: Consider changing this identifier
                lane3d_results_dict["identifier"] = (
                    # img_metas[0]["split"][0],
                    img_metas[0]["scene_token"],
                    img_metas[0]["sample_idx"],
                )
                lane3d_results_dict["lane_lines"] = lane_lines
                lane3d_results.append(lane3d_results_dict)

        return bbox_results, lane3d_results

    # TODO：Change here to support multi-scale testing on 3dlane
    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        if isinstance(img, mmcv.parallel.data_container.DataContainer):
            img = img.data[0].to("cuda")
            img_metas = img_metas.data[0]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        result_list = [dict() for i in range(len(img_metas))]
        bbox_pts, lane3d_results = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale
        )
        # for result_dict, pts_bbox, lane3d_result in zip(result_list, bbox_pts):
        for result_dict, pts_bbox, lane_3d, metas in zip(
            result_list, bbox_pts, lane3d_results, img_metas
        ):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["lane3d"] = lane_3d
            if "scene_token" in metas:
                result_dict["scene_token"] = metas["scene_token"]
            if "sample_idx" in metas:
                result_dict["sample_idx"] = metas["sample_idx"]
        return result_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list
