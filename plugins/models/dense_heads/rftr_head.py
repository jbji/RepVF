import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.core import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils import NormedLinear, build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from openlanev1.utils import *
from openlanev1.visualization import show_2d_bbox, show_2d_full, adapt_results
from mmcv.cnn import (
    Conv2d,
    Linear,
    bias_init_with_prob,
    build_activation_layer,
    constant_init,
    kaiming_init,
    xavier_init,
)
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from ...core.bbox.util import normalize_bbox


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


@HEADS.register_module()
class RFTRHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(
        self,
        in_channels,
        num_fcs=2,
        transformer=None,
        sync_cls_avg_factor=False,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        code_weights=None,
        r_dict=dict(
            num_queries_sets_task=[100, 100],
            num_points_per_set=10,
            set_point_embed_dims=16,
            normedlinear=False,
        ),
        bbox3d_params_dict=dict(
            num_classes_bbox=3,
            bbox_coder=None,
            loss_cls=dict(
                type="CrossEntropyLoss",
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
            loss_ptb=dict(loss_weight=1e-4, gt_scalar=20.0),
        ),
        lane3d_params_dict=dict(
            max_num=30,
            num_lane_cls=21,
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.05,
            ),
            loss_reg=dict(type="L1Loss", loss_weight=0.003),
        ),
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            )
        ),
        test_cfg=dict(max_per_img=100),
        with_position=True,
        with_multiview=False,
        depth_step=0.8,
        depth_num=64,
        LID=False,
        depth_start=1,
        position_range=[-65, -65, -8.0, 65, 65, 8.0],
        init_cfg=None,
        normedlinear=False,
        **kwargs,
    ):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[: self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        loss_bbox_cls = bbox3d_params_dict["loss_cls"]
        loss_bbox = bbox3d_params_dict["loss_bbox"]
        loss_bbox_iou = bbox3d_params_dict["loss_iou"]
        num_classes_bbox = bbox3d_params_dict["num_classes_bbox"]
        class_weight = loss_bbox_cls.get("class_weight", None)
        if class_weight is not None and issubclass(self.__class__, PETRHead_uni):
            assert isinstance(class_weight, float), (
                "Expected "
                "class_weight to have type float. Found "
                f"{type(class_weight)}."
            )
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_bbox_cls.get("bg_cls_weight", class_weight)
            assert isinstance(bg_cls_weight, float), (
                "Expected "
                "bg_cls_weight to have type float. Found "
                f"{type(bg_cls_weight)}."
            )
            class_weight = torch.ones(num_classes_bbox + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes_bbox] = bg_cls_weight
            loss_bbox_cls.update({"class_weight": class_weight})
            if "bg_cls_weight" in loss_bbox_cls:
                loss_bbox_cls.pop("bg_cls_weight")
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert "assigner" in train_cfg, (
                "assigner should be provided " "when train_cfg is set."
            )
            assigner = train_cfg["assigner"]
            assert loss_bbox_cls["loss_weight"] == assigner["cls_cost"]["weight"], (
                "The classification weight for loss and matcher should be"
                "exactly the same."
            )
            assert loss_bbox["loss_weight"] == assigner["reg_cost"]["weight"], (
                "The regression L1 weight for loss and matcher "
                "should be exactly the same."
            )
            assigner_lane3d = train_cfg["assigner_lane3d"]
            # assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
            #     'The regression iou weight for loss and matcher should be' \
            #     'exactly the same.'
            self.assigner = build_assigner(assigner)
            self.assigner_lane3d = build_assigner(assigner_lane3d)
            self._init_assigner_dummy()
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type="PseudoSampler")
            # mmdet.core.bbox.samplers.pseudo_sampler, it directly returns the positive and negative indices
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_queries_sets_task = r_dict["num_queries_sets_task"]
        self.num_queries_sets = sum(self.num_queries_sets_task)
        self.num_points_per_set = r_dict["num_points_per_set"]
        self.set_point_embed_dims = r_dict["set_point_embed_dims"]
        self.lane3d_cls_out_channels = lane3d_params_dict["num_lane_cls"]

        self.in_channels = in_channels

        self.num_fcs = num_fcs

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256

        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview

        self.bbox3d_enabled = True
        self.lane3d_enabled = True

        if self.bbox3d_enabled:
            self.bbox3d_params_dict = bbox3d_params_dict
            self.num_classes_bbox = bbox3d_params_dict["num_classes_bbox"]
            self.loss_ptb_weight = bbox3d_params_dict["loss_ptb"]["loss_weight"]
            # self.loss_ptb_gt_scalar = bbox3d_params_dict["loss_ptb"]["gt_scalar"]

        if self.lane3d_enabled:
            self.lane3d_params_dict = lane3d_params_dict

        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
        )
        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU", inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear

        if loss_bbox_cls["use_sigmoid"]:
            self.bbox_cls_out_channels = num_classes_bbox
        else:
            self.bbox_cls_out_channels = num_classes_bbox + 1

        self._init_semantic_dim()
        self.lane3d_max_num = lane3d_params_dict["max_num"]

        # NOTE: normal parameter initialization must be put before this statement
        super().__init__(num_classes_bbox, in_channels, init_cfg=init_cfg)
        # but module assignments must be put here.

        self.loss_bbox_cls = build_loss(loss_bbox_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_iou = build_loss(loss_bbox_iou)

        # self.activate = build_activation_layer(self.act_cfg)
        # if self.with_multiview or not self.with_position:
        #     self.positional_encoding = build_positional_encoding(
        #         positional_encoding)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        # self.transformer_bbox3d = build_transformer(transformer)
        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )
        self.bbox_coder = build_bbox_coder(bbox3d_params_dict["bbox_coder"])
        # for 3d lane
        # self.transformer_lane3d = build_transformer(transformer_lane3d)
        self.loss_lane3d_cls = build_loss(self.lane3d_params_dict["loss_cls"])
        self.loss_lane3d_reg = build_loss(self.lane3d_params_dict["loss_reg"])
        # self.loss_lane3d_vis = build_loss(self.lane_params_dict["loss_vis"])
        self.loss_bbox_ang = build_loss(self.bbox3d_params_dict["loss_ang"])
        self.loss_lane3d_ang = build_loss(self.lane3d_params_dict["loss_ang"])
        # others after this statement
        self._before_init_layers()
        self._init_layers()

    def _init_semantic_dim(self):
        self.semantic_dim = (
            self.bbox_cls_out_channels
            * self.lane3d_cls_out_channels
            // math.gcd(self.bbox_cls_out_channels, self.lane3d_cls_out_channels)
        )  # 21 * 4 // 1 = 21*4 = 84

    def _init_assigner_dummy(self):
        pass

    def _before_init_layers(self):
        pass

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims * 3 // 2,
                    self.embed_dims * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    self.embed_dims * 4,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0
                ),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(
                    self.position_dim,
                    self.embed_dims * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    self.embed_dims * 4,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

        # for unified design
        # self.num_queries_sets = [100, 100]
        # self.num_points_per_set = 10
        self.reference_point_sets = nn.Embedding(
            self.num_queries_sets, 3 * self.num_points_per_set
        )
        # self.set_point_embed_dims = 16
        self.set_embed_dims = self.set_point_embed_dims * 2 * self.num_points_per_set
        self.query_embedding_sets = nn.Sequential(
            nn.Linear(self.set_embed_dims * 3 // 2, self.set_embed_dims),
            nn.ReLU(),
            nn.Linear(self.set_embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self._init_geometry_branch()
        self._init_semantic_branch()
        self._init_interpreters()

    def _init_geometry_branch(self, pred_size=5):
        geometry_branch = []
        for _ in range(self.num_fcs):
            geometry_branch.append(Linear(self.embed_dims, self.embed_dims))
            geometry_branch.append(nn.LayerNorm(self.embed_dims))
            geometry_branch.append(nn.ReLU(inplace=True))
        geometry_branch.append(
            Linear(self.embed_dims, self.num_points_per_set * pred_size)
        )
        shared_geometry_layer = nn.Sequential(*geometry_branch)
        self.geometry_branch = nn.ModuleList(
            [shared_geometry_layer for _ in range(self.num_pred)]
        )

    def _init_semantic_branch(self):
        semantic_branch = []
        for _ in range(self.num_fcs):
            semantic_branch.append(Linear(self.embed_dims, self.embed_dims))
            semantic_branch.append(nn.LayerNorm(self.embed_dims))
            semantic_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            semantic_branch.append(NormedLinear(self.embed_dims, self.semantic_dim))
        else:
            semantic_branch.append(Linear(self.embed_dims, self.semantic_dim))
        shared_semantic_layer = nn.Sequential(*semantic_branch)
        self.semantic_branch = nn.ModuleList(
            [shared_semantic_layer for _ in range(self.num_pred)]
        )

    def _init_interpreters(self):
        self.bbox3d_heading_calc = VectorField2DHeading()
        self.bbox3d_geometry_interpreter = DirectionVectorStdBBoxInterpreter()
        self.lane3d_geometry_interpreter = PointSortInterpreter()

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

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]["pad_shape"][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index = torch.arange(
                start=0, end=self.depth_num, step=1, device=img_feats[0].device
            ).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num)
            )
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(
                start=0, end=self.depth_num, step=1, device=img_feats[0].device
            ).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(
            1, 2, 3, 0
        )  # W, H, D, 3
        # why an extra 1? (W, H, D, 4) a: for homogeneous coordinate!
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        # why coord[..., :2] *= depth[..., 2]? best guess: to make sure the depth is not too small???
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps
        )

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            # u, v depth?
            for i in range(len(img_meta["lidar2img"])):
                img2lidar.append(np.linalg.inv(img_meta["lidar2img"][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0]
        )
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1]
        )
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2]
        )

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)  # [1, 1, 32, 88]
        coords3d = (
            coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        )  # [B, N, 88, 32, 64 D, 3] -> [B*N, 192, 32, 88]
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get("version", None)
        if (version is None or version < 2) and issubclass(
            self.__class__, PETRHead_uni
        ):
            convert_dict = {
                ".self_attn.": ".attentions.0.",
                # '.ffn.': '.ffns.0.',
                ".multihead_attn.": ".attentions.1.",
                ".decoder.norm.": ".decoder.post_norm.",
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _forward_decoding(self, outs_dec, reference_point_sets, batch_size):
        outs_semantic = []
        outs_geometry = []
        position_range = torch.tensor(self.position_range, dtype=torch.float32).cuda()
        position_range = position_range.view(2, 3)[:, [0, 1, 2]]  # Shape (2, 3)
        min_values, max_values = position_range[0], position_range[1]
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_point_sets.clone())
            _to_decode = outs_dec[lvl]
            g_out = self.geometry_branch[lvl](_to_decode)
            reference = reference.reshape(
                batch_size, self.num_queries_sets, self.num_points_per_set, 3
            )
            g_out = g_out.reshape(
                batch_size, self.num_queries_sets, self.num_points_per_set, -1
            )
            g_out[..., :3] += reference
            g_out[..., :3] = g_out[..., :3].sigmoid()
            g_out[..., :3] = g_out[..., :3] * (max_values - min_values) + min_values
            outs_geometry.append(g_out)
            s_out = self.semantic_branch[lvl](_to_decode)
            outs_semantic.append(s_out)
        return outs_geometry, outs_semantic

    def _forward_transformer_wrapper(self, x, masks, query_embeds_sets, pos_embed):
        outs_dec, _ = self.transformer(
            x, masks, query_embeds_sets, pos_embed, self.geometry_branch
        )
        outs_dec = torch.nan_to_num(outs_dec)
        return outs_dec

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[0]  # [1,1, 256, 32, 88], 1/16
        batch_size, num_cams = x.size(0), x.size(1)  # 1, 1
        input_img_h, input_img_w, _ = img_metas[0]["pad_shape"][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w)
        )  # [1,1,512,1408]
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]["img_shape"][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        x = self.input_proj(x.flatten(0, 1))  # 1/16 2d feature from backbone
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)  # [1,1,32,88]

        # position embeding and position encoding
        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(
                mlvl_feats, img_metas, masks
            )
            pos_embed = coords_position_embeding
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)  # [2,1,384,32,88]
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(
                    x.size()
                )  # conv2d
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        outs = {}

        reference_point_sets = self.reference_point_sets.weight
        query_embeds_sets = self.query_embedding_sets(
            pos2posemb3d(
                reference_point_sets.reshape(
                    self.num_queries_sets, self.num_points_per_set, 3
                ),
                num_pos_feats=self.set_point_embed_dims,
            ).reshape(
                self.num_queries_sets, -1
            )  # (bs, num_queries_sets, set_embed_dims*3//2)
        )

        reference_point_sets = reference_point_sets.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        self._save_qkv_hook(
            query_embeds_sets, pos_embed, x, masks, reference_point_sets, img_metas
        )
        # sum(bs,self.num_queries_sets, self.embed_dims)
        # shape [num_layers, bs, num_query_sets, dim]
        outs_dec = self._forward_transformer_wrapper(
            x, masks, query_embeds_sets, pos_embed
        )
        # reshape to [num_layers, bs, num_x_steps, num_y_steps, embed_dims]
        outs_dec = outs_dec.reshape(
            -1,
            batch_size,
            self.num_queries_sets,
            self.embed_dims,
        )
        outs_geometry, outs_semantic = self._forward_decoding(
            outs_dec, reference_point_sets, batch_size
        )

        # [6,bs,200,10,3]
        all_outs_geometry = torch.stack(outs_geometry)
        # [6,bs,200,30]
        all_outs_semantic = torch.stack(outs_semantic)

        (
            bbox3d_geometry,
            bbox3d_semantic,
            lane3d_geometry,
            lane3d_semantic,
        ) = self._parameter_free_transduction(all_outs_geometry, all_outs_semantic)

        self._save_for_visualization_hook(
            img_metas,
            all_outs_geometry,
            all_outs_semantic,
            bbox3d_geometry,
            bbox3d_semantic,
            lane3d_geometry,
            lane3d_semantic,
        )
        self._save_for_plt_hook(
            img_metas,
            all_outs_geometry,
            all_outs_semantic,
            bbox3d_geometry,
            bbox3d_semantic,
            lane3d_geometry,
            lane3d_semantic,
        )

        outs.update(
            {
                "all_outs_geometry": all_outs_geometry,
                "all_outs_semantic": all_outs_semantic,
                "all_cls_scores": bbox3d_semantic,
                "all_bbox_preds": bbox3d_geometry,
                "all_lane3d_preds_xyz": lane3d_geometry,
                "all_lane3d_cls_scores": lane3d_semantic,
                "all_lane3d_vis_scores": None,
            }
        )

        return outs

    def _save_qkv_hook(
        self, query_embeds_sets, pos_embed, x, masks, reference_point_sets, img_metas
    ):
        if os.environ.get("SAVE_QKV") == "True":
            # for center point query
            query_embeds_center = self.query_embedding_sets(
                pos2posemb3d(
                    reference_point_sets.reshape(
                        self.num_queries_sets, self.num_points_per_set, 3
                    ).mean(1, keepdim=True),
                    num_pos_feats=self.set_point_embed_dims,
                )
                .repeat(1, self.num_points_per_set, 1)
                .reshape(
                    self.num_queries_sets, -1
                )  # (bs, num_queries_sets, set_embed_dims*3//2)
            )
            npdump_nocheck(
                out_type="query_embeds_center",
                np_obj=query_embeds_center.cpu().numpy(),
                img_metas=img_metas,
            )
            npdump_nocheck(
                out_type="query_embeds_sets",
                np_obj=query_embeds_sets.cpu().numpy(),
                img_metas=img_metas,
            )
            npdump_nocheck(
                out_type="pos_embed",
                np_obj=pos_embed.cpu().numpy(),
                img_metas=img_metas,
            )
            npdump_nocheck(
                out_type="x",
                np_obj=x.cpu().numpy(),
                img_metas=img_metas,
            )
            npdump_nocheck(
                out_type="masks",
                np_obj=masks.cpu().numpy(),
                img_metas=img_metas,
            )
            npdump_nocheck(
                out_type="reference_point_sets",
                np_obj=reference_point_sets.cpu().numpy(),
                img_metas=img_metas,
            )

    def _save_for_visualization_hook(
        self,
        img_metas,
        all_outs_geometry,
        all_outs_semantic,
        bbox3d_geometry,
        bbox3d_semantic,
        lane3d_geometry,
        lane3d_semantic,
    ):
        if os.environ.get("SAVE_FOR_VISUALIZATION") == "True":
            out_types = [
                "bbox3d_geometry",
                "bbox3d_semantic",
                "lane3d_geometry",
                "lane3d_semantic",
                "all_outs_geometry",
                "all_outs_semantic",
            ]
            objs = [
                bbox3d_geometry,
                bbox3d_semantic,
                lane3d_geometry,
                lane3d_semantic,
                all_outs_geometry,
                all_outs_semantic,
            ]
            for type_i, obj_i in zip(out_types, objs):
                npdump(
                    out_type=type_i,
                    np_obj=obj_i.cpu().numpy(),
                    img_metas=img_metas,
                )

    def _save_for_plt_hook(
        self,
        img_metas,
        all_outs_geometry,
        all_outs_semantic,
        bbox3d_geometry,
        bbox3d_semantic,
        lane3d_geometry,
        lane3d_semantic,
    ):
        custom_suffix = os.environ.get("SAVE_PLT_BBOX_SUFFIX")
        if custom_suffix is None:
            custom_suffix = "threshold_0.5"

        if os.environ.get("SAVE_PLT_BBOX") == "True":
            if int(img_metas[0]["sample_idx"]) // 100 % 50 == 0:
                fbs, fbg, fbg_p, fls, flg = adapt_results(
                    all_outs_geometry.cpu().numpy()[-1, 0],
                    bbox3d_geometry.cpu().numpy()[-1, 0],
                    lane3d_geometry.cpu().numpy()[-1, 0],
                    bbox3d_semantic.cpu().numpy()[-1, 0],
                    lane3d_semantic.cpu().numpy()[-1, 0],
                    ks=[300, 15],
                    query_nums=self.num_queries_sets_task,
                    bbox3d_threshold=0.5,
                    lane3d_threshold=0.5,
                )
                cameras_path = img_metas[0]["img_paths"]
                # intrinsics, extrinsics = None, None
                intrinsics = img_metas[0]["calib_original"]["intrinsics"]
                extrinsics = img_metas[0]["calib_original"]["extrinsics"]
                # lidar2img = img_metas[0]["lidar2img"]
                # show_2d_bbox(
                #     img_metas[0]["sample_idx"],
                #     cameras_path,
                #     intrinsics,
                #     extrinsics,
                #     fbg,
                #     fbg_p,
                #     with_gt=True,
                #     custom_suffix="threshold_0.2",
                #     # lidar2img,
                # )
                show_2d_full(
                    img_metas[0]["sample_idx"],
                    cameras_path,
                    intrinsics,
                    extrinsics,
                    fbg,
                    fbg_p,
                    fbs,
                    flg,
                    fls,
                    with_gt=False,
                    custom_suffix=custom_suffix,
                    show_vector=True,
                    vector_length=0.5,
                    # save_folder_root=result_folder,
                )

                out_types = [
                    "filtered_bbox_semantic",
                    "filtered_bbox3d_geometry",
                    "filtered_bbox3d_points",
                    "filtered_lane_semantic",
                    "filtered_lane_geometry",
                ]
                objs = [fbs, fbg, fbg_p, fls, flg]
                for type_i, obj_i in zip(out_types, objs):
                    npdump_nocheck(
                        out_type=type_i,
                        np_obj=obj_i,
                        img_metas=img_metas,
                    )

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

    def _get_target_single(
        self, cls_score, bbox_pred, gt_labels, gt_bboxes, gt_bboxes_ignore=None
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full(
            (num_bboxes,), self.num_classes_bbox, dtype=torch.long
        )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single_lane3d(
        self,
        preds_xyz,
        cls_scores,
        # vis_scores,
        gt_xyz,
        gt_cls,
        # gt_vis,
    ):
        num_lanes = preds_xyz.size(0)
        # assigner and sampler
        # The assigner should now also consider visibility scores and ground truth visibility
        assign_result = self.assigner_lane3d.assign(
            preds_xyz, cls_scores, gt_xyz, gt_cls  # , vis_scores, gt_vis
        )
        sampling_result = self.sampler.sample(assign_result, preds_xyz, gt_xyz)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_xyz.new_full(
            (num_lanes,),
            self.lane3d_params_dict["num_lane_cls"],
            dtype=torch.long,
        )
        labels[pos_inds] = gt_cls[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_xyz.new_ones(num_lanes)

        # visibility targets (new)
        # You might want to define visibility targets similar to how you define label targets
        # vis = gt_vis.new_zeros((num_lanes, self.num_points_per_set))
        # vis[pos_inds] = gt_vis[sampling_result.pos_assigned_gt_inds]
        # vis_weights = gt_vis.new_ones((num_lanes, self.num_points_per_set))

        # bbox (lane points in your case) targets
        # code_size = gt_xyz.size(1)
        lane_targets = torch.zeros_like(preds_xyz)  #   [..., :code_size]
        lane_weights = torch.zeros_like(preds_xyz)
        lane_weights[pos_inds] = 1.0
        lane_targets[pos_inds] = sampling_result.pos_gt_bboxes

        # Return visibility targets and weights in addition to existing outputs
        return (
            labels,
            label_weights,
            lane_targets,
            lane_weights,
            # vis,
            # vis_weights,
            pos_inds,
            neg_inds,
        )

    def get_targets_lane3d(
        self,
        preds_xyz_list,
        cls_scores_list,
        # vis_scores_list,
        gt_xyz_list,
        gt_cls_list,
        # gt_vis_list,
    ):
        (
            labels_list,
            label_weights_list,
            lane_targets_list,
            lane_weights_list,
            # vis_list,
            # vis_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single_lane3d,
            preds_xyz_list,
            cls_scores_list,
            # vis_scores_list,
            gt_xyz_list,
            gt_cls_list,
            # gt_vis_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            lane_targets_list,
            lane_weights_list,
            # vis_list,
            # vis_weights_list,
            num_total_pos,
            num_total_neg,
        )

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

    def loss_single_point_to_bbox(self, point_preds, bbox_preds):
        """
        Calculate the distance from points to the closest face of the bounding box.

        Args:
            bbox_preds: Tensor of shape (bs, num_query, 10) containing the bounding box predictions, in the format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
            Note that x y z cooresponds to l w h respectively.
            point_preds: Tensor of shape (bs, num_query, num_points_per_set, 3) containing the point predictions, in the format (x, y, z)
        Returns:
            Tensor: The sum of distances from all points to the bounding box
        """
        point_preds = point_preds.reshape(
            -1, point_preds.size(-2), point_preds.size(-1)
        )
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        cx, cy, cz = bbox_preds[..., 0], bbox_preds[..., 1], bbox_preds[..., 4]
        l, w, h = bbox_preds[..., 3], bbox_preds[..., 2], bbox_preds[..., 5]
        rot_sine, rot_cosine = bbox_preds[..., 6], bbox_preds[..., 7]

        box_center = torch.stack([cx, cy, cz], dim=-1)
        box_size = torch.stack([l, w, h], dim=-1)
        bbox_min = box_center - box_size / 2
        bbox_max = box_center + box_size / 2

        theta = torch.atan2(rot_sine, rot_cosine)
        identity_3x3 = torch.eye(3, device=theta.device).repeat(*theta.shape, 1, 1)

        cos_theta = torch.cos(-theta)
        sin_theta = torch.sin(-theta)

        rotation_matrix_neg_theta = identity_3x3.clone()  # Start with identity matrices
        rotation_matrix_neg_theta[..., 0, 0] = cos_theta
        rotation_matrix_neg_theta[..., 0, 1] = -sin_theta
        rotation_matrix_neg_theta[..., 1, 0] = sin_theta
        rotation_matrix_neg_theta[..., 1, 1] = cos_theta
        point_preds = torch.matmul(
            point_preds, rotation_matrix_neg_theta.transpose(-1, -2)
        )

        # Normalize point_preds and bbox max/min
        position_range = torch.tensor(self.position_range, dtype=torch.float32).to(
            device=point_preds.device
        )
        position_range = position_range.view(2, 3)[:, [0, 1, 2]]  # Shape (2, 3)
        min_values, max_values = position_range[0], position_range[1]
        bbox_min = (bbox_min - min_values) / (max_values - min_values)
        bbox_max = (bbox_max - min_values) / (max_values - min_values)
        point_preds = (point_preds - min_values) / (max_values - min_values)

        # Calculate distances in each dimension
        d_x = torch.clamp(
            torch.max(
                bbox_min[..., 0].unsqueeze(-1) - point_preds[..., 0],
                point_preds[..., 0] - bbox_max[..., 0].unsqueeze(-1),
            ),
            min=0,
        )
        d_y = torch.clamp(
            torch.max(
                bbox_min[..., 1].unsqueeze(-1) - point_preds[..., 1],
                point_preds[..., 1] - bbox_max[..., 1].unsqueeze(-1),
            ),
            min=0,
        )
        d_z = torch.clamp(
            torch.max(
                bbox_min[..., 2].unsqueeze(-1) - point_preds[..., 2],
                point_preds[..., 2] - bbox_max[..., 2].unsqueeze(-1),
            ),
            min=0,
        )
        # Sum up the distances to get the final distance for each point
        d = d_x + d_y + d_z

        # Sum the loss over all points
        loss = torch.nan_to_num(torch.sum(d) * self.loss_ptb_weight)

        return loss, None

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_lanes_3d,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        """ "Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
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

        if self.lane3d_enabled:
            loss_dict_lane3d = self.loss_lane3d(gt_lanes_3d, preds_dicts)
            loss_dict.update(loss_dict_lane3d)

        return loss_dict

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

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # pred_dicts = {
        # "all_cls_scores": preds_dicts["all_cls_scores"],
        # "all_bbox_preds": preds_dicts["all_bbox_preds"],
        # }
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]["box_type_3d"](bboxes, bboxes.size(-1))
            scores = preds["scores"]
            labels = preds["labels"]
            ret_list.append([bboxes, scores, labels])

        # if os.environ.get("SAVE_FOR_VISUALIZATION") == "True":
        # np_obj = np.array(
        # [[ret[0].tensor.cpu(), ret[1].cpu(), ret[2].cpu()] for ret in ret_list]
        # )
        # npdump(out_type="bbox3d", np_obj=np_obj, img_metas=img_metas)

        return ret_list

    @force_fp32(apply_to=("preds_dicts"))
    def get_lanes(self, preds_dicts, img_metas, rescale=False):
        """Generate 3dlane from 3dlane head predictions.
        Args: ommited
        return: preds by batch
        """

        preds_xyz = preds_dicts["all_lane3d_preds_xyz"][..., :3].detach().cpu().numpy()
        cls_scores = torch.sigmoid(preds_dicts["all_lane3d_cls_scores"])
        cls_scores = cls_scores.detach().cpu().numpy()

        preds_by_batch = []
        batch_size = preds_xyz.shape[1]
        layer_count = preds_xyz.shape[0]
        for bs_j in range(batch_size):
            layer_preds = []
            for layer_i in range(layer_count):
                pred_xyz = preds_xyz[layer_i][bs_j]
                cls_score = cls_scores[layer_i][bs_j]
                # vis_score = vis_scores[layer_i][bs_j]

                # Sort and pick top-k class scores
                aggregate_scores = np.max(cls_score, axis=-1)
                sorted_indices = np.argsort(aggregate_scores)[
                    ::-1
                ]  # Sort in descending order
                k = self.lane3d_max_num  # Pick top-k indices
                topk_indices = sorted_indices[:k]  # Pick top-k indices
                topk_scores = cls_score[topk_indices]  # Get top-k scores
                topk_xyz = pred_xyz[topk_indices]  # Get top-k xyz

                layer_preds.append((topk_xyz.tolist(), topk_scores.tolist()))
            preds_by_batch.append(layer_preds)
        # npdump(out_type="lane3d", np_obj=np.array(preds_by_batch), img_metas=img_metas)
        return preds_by_batch
