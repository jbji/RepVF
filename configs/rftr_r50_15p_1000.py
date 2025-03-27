custom_imports = dict(imports=["projects.repvf.plugins"])

_base_ = [
    "./_base_/datasets/unified_waymo_1000_15p.py",
    "./_base_/default_runtime.py",
]

merge_road_edge = True

# SyncBN = True

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
        positional_encoding=dict(
            type="SinePositionalEncoding3D", num_feats=128, normalize=True
        ),
        r_dict=dict(  # r for representation
            num_queries_sets_task=[600, 150],
            num_points_per_set=15,
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


# vibrant-blaze-62
# 46566 2024-03-02 17:10:13,200 - mmdet - INFO - ===> Evaluation on validation set:
# 46567 laneline F-measure 0.61758979
# 46568 laneline Recall  0.56823237
# 46569 laneline Precision  0.67633736
# 46570 laneline Category Accuracy  0.91631044
# 46571 laneline x error (close)  0.34140357 m
# 46572 laneline x error (far)  0.44968619 m
# 46573 laneline z error (close)  0.072532891 m
# 46574 laneline z error (far)  0.1070766 m
# 46575 2024-03-02 17:10:13,221 - mmdet - INFO - Exp name: 1000_15p.py
# 46576 2024-03-02 17:10:13,229 - mmdet - INFO - Epoch(val) [24][9797] Vehicle/L1 mAP: 0.2263, Vehicle/L1 mAPH: 0.2239, Vehicle/L2 mAP: 0.0192, Vehicle/L2 mAPH: 0.0189, Pedestrian/L1 mAP: 0.0803, Pedestrian/L1 mAPH: 0.0724, Pedestrian/L2 mAP: 0.0021, Pedestrian/L2 mAPH: 0.0019, Sign/L1 mAP: 0.0000, Sign/L1 mAPH: 0.0000, Sign/L2 mAP: 0.0000, Sign/L2 mAPH: 0.0000, Cyclist/L1 mAP: 0.2776, Cyclist/L1 mAPH: 0.2631, Cyclist/L2 mAP: 0.0107, Cyclist/L2 mAPH: 0.0101, Overall/L1 mAP: 0.1947, Overall/L1 mAPH: 0.1865, Overall/L2 mAP: 0.0107, Overall/L2 mAPH: 0.0103, laneline F-measure: 0.6176, laneline Recall: 0.5682, laneline Precision: 0.6763, laneline Category Accuracy: 0.9163, laneline x error (close): 0.3414, laneline x error (far): 0.4497, laneline z error (close): 0.0725, laneline z error (far): 0.1071
# 46577 eval:39001/39188
# 46578 eval:39188/39188
