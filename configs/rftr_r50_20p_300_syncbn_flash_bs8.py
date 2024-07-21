custom_imports = dict(imports=["projects.uni.plugins"])

_base_ = [
    "./_base_/datasets/unified_waymo_300_20p_bs8.py",
    "./_base_/default_runtime.py",
]

merge_road_edge = True

SyncBN = True  # this need a modification to train.py

point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]  # for waymo
label_range = [-84.88, -84.88, -10.0, 84.88, 84.88, 10.0]
voxel_size = [0.2, 0.2, 8]
backbone_norm_cfg = dict(type="LN", requires_grad=True)
num_classes_bbox = 3
model = dict(
    type="RFTR",
    use_grid_mask=True,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(
            2,
            3,
        ),
        frozen_stages=-1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        with_cp=True,
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        pretrained="ckpts/resnet50_msra-5891d200.pth",
    ),
    img_neck=dict(type="CPFPN", in_channels=[1024, 2048], out_channels=256, num_outs=2),
    pts_bbox_head=dict(
        type="RFTRHead",
        in_channels=256,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=label_range,
        transformer=dict(
            type="PETRTransformer",
            decoder=dict(
                type="PETRTransformerDecoder",
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type="PETRTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="PETRMultiheadFlashAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding3D", num_feats=128, normalize=True
        ),
        r_dict=dict(  # r for representation
            num_queries_sets_task=[600, 150],
            num_points_per_set=20,
            set_point_embed_dims=16,
            normedlinear=False,
        ),
        code_size=8,
        bbox3d_params_dict=dict(
            num_classes_bbox=num_classes_bbox,
            bbox_coder=dict(
                type="NMSFreeCoder",
                # type='NMSFreeClsCoder',
                post_center_range=label_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=num_classes_bbox,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.1),
            loss_ang=dict(type="L1Loss", loss_weight=0.2),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
            loss_ptb=dict(loss_weight=1e-4),
        ),
        lane3d_params_dict=dict(
            max_num=30,
            num_lane_cls=21 if merge_road_edge else 22,
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(type="L1Loss", loss_weight=0.004),
            loss_ang=dict(type="L1Loss", loss_weight=0.032),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3DCenter",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.1),
                iou_cost=dict(type="IoUCost", weight=0.0),
                pc_range=point_cloud_range,
            ),
            assigner_lane3d=dict(
                type="HungarianAssignerLane3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="Lane3DL1Cost", weight=0.003),
            ),
        )
    ),
)


find_unused_parameters = False
load_from = None
resume_from = None

# specifically for flash attention and syncbn

optimizer_config = dict(
    type="Fp16OptimizerHook",
    loss_scale="dynamic",
    grad_clip=dict(max_norm=35, norm_type=2),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.25),  # let's see if it can converge
        }
    ),
    weight_decay=0.01,
)


# incandescent-monkey-45
# 3562 2024-02-24 06:55:38,398 - mmdet - INFO - ===> Evaluation on validation set:
# 3563 laneline F-measure 0.66546884
# 3564 laneline Recall  0.58934647
# 3565 laneline Precision  0.76417244
# 3566 laneline Category Accuracy  0.91719289
# 3567 laneline x error (close)  0.4201351 m
# 3568 laneline x error (far)  0.50179366 m
# 3569 laneline z error (close)  0.10345165 m
# 3570 laneline z error (far)  0.13054209 m
# 3571 2024-02-24 06:55:38,403 - mmdet - INFO - Exp name: 20p_syncbn_flash_bs8.py
# 3572 2024-02-24 06:55:38,403 - mmdet - INFO - Epoch(val) [24][2824] Vehicle/L1 mAP: 0.2532, Vehicle/L1 mAPH: 0.2503, Vehicle/L2 mAP: 0.0106, Vehicle/L2 mAPH: 0.0105, Pedestrian/L1 mAP: 0.0149, Pedestrian/L1 mAPH: 0.0096, Pedestrian/L2 mAP: 0.0001, Pedestrian/L2 mAPH: 0.0000, Sign/L1 mAP: 0.0000, Sign/L1 mAPH: 0.0000, Sign/L2 mAP: 0.0000, Sign/L2 mAPH: 0.0000, Cyclist/L1 mAP: 0.0167, Cyclist/L1 mAPH: 0.0164, Cyclist/L2 mAP: 0.0000, Cyclist/L2 mAPH: 0.0000, Overall/L1 mAP: 0.0949, Overall/L1 mAPH: 0.0921, Overall/L2 mAP: 0.0036, Overall/L2 mAPH: 0.0035, laneline F-measure: 0.6655, laneline Recall: 0.5893, laneline Precision: 0.7642, laneline Category Accuracy: 0.9172, laneline x error (close): 0.4201, laneline x error (far): 0.5018, laneline z error (close): 0.1035, laneline z error (far): 0.1305
