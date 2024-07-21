_base_ = [
    "../default_optimizer.py",
]

point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]  # for waymo

file_client_args = dict(backend="disk")
db_sampler = dict()

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
pi = 3.1415927
# aligned with openlane
ida_aug_conf = {
    "resize_lim": (1, 1),
    "final_dim": (
        512,
        1408,
    ),
    "bot_pct_lim": (
        0.0,
        0.0,
    ),
    "rot_lim": (
        -pi / 18,
        pi / 18,
    ),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}
# For waymo we usually do 3-class detection
class_names = ["Vehicle", "Pedestrian", "Cyclist"]
train_pipeline = [
    dict(type="CustomLoadImagesFromFile", to_float32=True),
    dict(type="UpdateCameraParameters"),
    # 3D Object Detection
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    # 3D Lane Detection
    dict(
        type="LoadAnnotations3DLane",
        camera_frustum_to_world=True,
        apollo_to_openlane=False,
        merge_road_edge=True,
    ),
    dict(
        type="Lane3DRangeVisFilter",
        by_visibility=True,
        visibility_threshold=0,
        by_range=True,
        y_range=(-10, 10, 3),
        x_range_to_prune=(3, 84.88),
    ),
    dict(type="Lane3DResampler", num_points=15),
    # Augmentations
    dict(
        type="CustomResizeCropFlipImage",
        fixed_resize=(900, 1600),  # resize to adapt into this network.
        data_aug_conf=ida_aug_conf,
        training=True,
        for_bbox=True,
    ),
    # Update: img, calib-intrinsic
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),  # backbone is different
    # Update: img
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="CustomFormatBundle3D", class_names=class_names
    ),  # format image to (N, 1, C, H, W)
    dict(
        type="Collect3D",
        keys=[
            "img",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_lanes_3d",
        ],  # , 'gt_anchor_lanes'
        meta_keys=[
            "scene_token",
            "sample_idx",
            "img_paths",
            "img_shape",
            "pad_shape",
            "lidar2img",  # , 'scale_factor', 'pad_shape',
            #'lidar2img', 'can_bus',
        ],
    ),
]
test_pipeline = [
    dict(type="CustomLoadImagesFromFile", to_float32=True),
    # Add: img, img_shape, Update: None
    dict(type="UpdateCameraParameters"),
    # Add: gt_cam_height, gt_cam_pitch, Update: calib - intrinsic, extrinsic
    dict(
        type="CustomResizeCropFlipImage",
        fixed_resize=(900, 1600),
        data_aug_conf=ida_aug_conf,
        training=False,
        for_bbox=True,
    ),
    # Update: img, calib-intrinsic
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # Update: img
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="CustomFormatBundle3D", class_names=class_names, with_label=False
            ),  # format image to (N, 1, C, H, W)
            dict(
                type="Collect3D",
                keys=["img"],
                meta_keys=[
                    # "gt_cam_height",
                    # "split",
                    "scene_token",
                    "sample_idx",
                    "img_paths",
                    "img_shape",
                    "pad_shape",
                    "lidar2img",  # , 'scale_factor', 'pad_shape',
                    #'lidar2img', 'can_bus',
                    "box_type_3d",
                    "calib_original",  # add this before doing visualization
                ],
            ),
        ],
    ),
]
dataset_type = "UnifiedWaymoDataset"
data_root = "data/waymo/openlane_format"
collection_suffix = "filtered"
bin_name = "cam_gt_filtered.bin"  # only for test, please use "cam_gt_filtered.bin"
eval_kit_root = "RepVF/tools/waymo_utils/"  # "data/waymo/waymo-od/src/bazel-bin/waymo_open_dataset/metrics/tools/"
data = dict(
    samples_per_gpu=2,  # 2, 2:7.4G~ 4:13G~
    workers_per_gpu=4,  # 4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        collection=(
            "training" if collection_suffix is None else f"training_{collection_suffix}"
        ),
        pipeline=train_pipeline,
        test_mode=False,
        box_type_3d="LiDAR",
        with_velocity=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        collection=(
            "validation"
            if collection_suffix is None
            else f"validation_{collection_suffix}"
        ),
        pipeline=test_pipeline,
        test_mode=True,
        bbox_metric="mAP",
        bin_name=bin_name,
        det_eval_kit_root=eval_kit_root,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        collection=(
            "validation"
            if collection_suffix is None
            else f"validation_{collection_suffix}"
        ),
        pipeline=test_pipeline,
        test_mode=True,
        bbox_metric="mAP",
        bin_name=bin_name,
        det_eval_kit_root=eval_kit_root,
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

total_epochs = 24
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
evaluation = dict(interval=24, pipeline=test_pipeline)
