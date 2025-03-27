custom_imports = dict(imports=["projects.repvf.plugins"])

_base_ = [
    "./_base_/datasets/waymo_1000_multiview_detonly.py",
    "./_base_/default_runtime.py",
]
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
label_range = [-84.88, -84.88, -10.0, 84.88, 84.88, 10.0]
voxel_size = [0.2, 0.2, 8]
model = dict(
    type="Detr3D",
    use_grid_mask=True,
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type="Detr3DHead",
        num_query=900,
        num_classes=3,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=8,
        transformer=dict(
            type="Detr3DTransformer",
            decoder=dict(
                type="Detr3DTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="Detr3DCrossAtten",
                            pc_range=point_cloud_range,
                            num_points=1,
                            embed_dims=256,
                            num_cams=5,
                        ),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
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
            post_center_range=label_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=3,
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5
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

# download here: https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing, from https://github.com/WangYueFt/detr3d?tab=readme-ov-file
load_from = "ckpts/fcos3d.pth"
resume_from = None

# bumbleberry-flan-79
# [>>>>>>>>>>>>>>>>>>>>>>>] 39188/39188, 19.5 task/s, elapsed: 2012s, ETA:     0sunified_perception/tools/waymo_utils/compute_detection_metrics_main /tmp/tmpirly80af/results.bin data/waymo/openlane_format/cam_gt_filtered.bin
# {'Vehicle/L1 mAP': 0.0919949, 'Vehicle/L1 mAPH': 0.0908848, 'Vehicle/L2 mAP': 0.00252907, 'Vehicle/L2 mAPH': 0.00249539, 'Pedestrian/L1 mAP': 0.0528616, 'Pedestrian/L1 mAPH': 0.040642, 'Pedestrian/L2 mAP': 0.000999772, 'Pedestrian/L2 mAPH': 0.000749414, 'Sign/L1 mAP': 0.0, 'Sign/L1 mAPH': 0.0, 'Sign/L2 mAP': 0.0, 'Sign/L2 mAPH': 0.0, 'Cyclist/L1 mAP': 0.157895, 'Cyclist/L1 mAPH': 0.1553, 'Cyclist/L2 mAP': 0.00105416, 'Cyclist/L2 mAPH': 0.000959599, 'Overall/L1 mAP': 0.10091716666666667, 'Overall/L1 mAPH': 0.09560893333333333, 'Overall/L2 mAP': 0.0015276673333333333, 'Overall/L2 mAPH': 0.0014014676666666665}
