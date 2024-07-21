custom_imports = dict(imports=["projects.uni.plugins"])

_base_ = [
    "./_base_/datasets/waymo_1000_multiview_detonly.py",
    "./_base_/default_runtime.py",
]
backbone_norm_cfg = dict(type="LN", requires_grad=True)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]  # for waymo
label_range = [-84.88, -84.88, -10.0, 103, 84.88, 10.0]
voxel_size = [0.2, 0.2, 8]

model = dict(
    type="Petr3D",
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
        pretrained="open-mmlab://detectron2/resnet50_caffe",
    ),
    img_neck=dict(type="CPFPN", in_channels=[1024, 2048], out_channels=256, num_outs=2),
    pts_bbox_head=dict(
        type="PETRHead",
        num_classes=3,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=label_range,
        normedlinear=False,
        code_size=8,
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
                            type="PETRMultiheadAttention",
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
        bbox_coder=dict(
            type="NMSFreeCoder",
            # type='NMSFreeClsCoder',
            post_center_range=label_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=3,
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding3D", num_feats=128, normalize=True
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)


find_unused_parameters = False
load_from = None
resume_from = None

# pious-galaxy-68
# 6523 2024-03-06 12:11:40,042 - mmdet - INFO - Exp name: petr_r50dcn_gridmask_p4.py
# 46524 2024-03-06 12:11:40,043 - mmdet - INFO - Epoch(val) [24][9797] Vehicle/L1 mAP: 0.3111, Vehicle/L1 mAPH: 0.3080, Vehicle/L2 mAP: 0.0289, Vehicle/L2 mAPH: 0.0286, Pedestrian/L1 mAP: 0.1210, Pedestrian/L1 mAPH: 0.1057, Pedestrian/L2 mAP: 0.0055, Pedestrian/L2 mAPH: 0.0047, Sign/L1 mAP: 0.0000, Sign/L1 mAPH: 0.0000, Sign/L2 mAP: 0.0000, Sign/L2 mAPH: 0.0000, Cyclist/L1 mAP: 0.1950, Cyclist/L1 mAPH: 0.1764, Cyclist/L2 mAP: 0.0096, Cyclist/L2 mAPH: 0.0086, Overall/L1 mAP: 0.2090, Overall/L1 mAPH: 0.1967, Overall/L2 mAP: 0.0147, Overall/L2 mAPH: 0.0140
