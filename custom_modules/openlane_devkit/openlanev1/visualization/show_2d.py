import os

import cv2
import matplotlib.patches as patches
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from openlanev1.io import io

from .auxiliary import (
    draw_bbox,
    draw_bev_bbox,
    draw_point_cloud,
    draw_range_circle,
    get_8_corners_kitti,
    get_8_corners_kitti_gt,
    project_to_image,
    project_to_image_lidar2img,
)


def show_2d_bbox(
    sample_idx,
    cameras_path,
    intrinsics,
    extrinsics,
    fbg,
    fbg_p,
    with_gt=False,
    custom_suffix=None,
    save_folder_root="visualizations",
):
    # fig, axes = plt.subplots(
    #     2,
    #     3,
    #     figsize=(57.6, 22.4),
    #     gridspec_kw={"width_ratios": [1.92, 1.92, 1.92], "height_ratios": [1.28, 0.96]},
    # )  # Adjust figsize for higher resolution

    fig = plt.figure(
        figsize=(57.6, 21.66)
    )  # 22.4 # Adjust figsize for higher resolution
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.92, 1.92, 0.96, 0.96],
        height_ratios=[1.28, 0.886],
    )
    axes = [
        fig.add_subplot(p)
        for p in [gs[0, 0], gs[0, 1], gs[0, 2:], gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]]
    ]

    if with_gt:
        json_path = (
            cameras_path[0]
            .replace("images_0", "detection3d_1000")
            .replace(".jpg", ".json")
        )
        anno = io.json_load(json_path)

    # camera i
    index_map = [1, 0, 2, 3, 4]
    for i in range(5):
        # Draw the 3D bounding box on the image using the projected points
        image = cv2.imread(cameras_path[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.shape)
        intrinsic = intrinsics[i]
        extrinsic = extrinsics[i]
        if with_gt:
            for bbox in anno["bbox"]:
                # bbox = annotation['bbox'][15]
                points_image = project_to_image(
                    get_8_corners_kitti_gt(bbox).T, intrinsic, extrinsic
                )
                if not isinstance(points_image, np.ndarray):
                    continue
                # print(points_image)
                image = draw_bbox(image, points_image, color=(0, 0, 255))
                # break

        for bbox in fbg:
            # bbox = annotation['bbox'][15]
            points_image = project_to_image(
                get_8_corners_kitti(bbox).T, intrinsic, extrinsic
            )
            if not isinstance(points_image, np.ndarray):
                continue
            # print(points_image)
            image = draw_bbox(image, points_image, color=(255, 0, 0))
            # break

        ai = index_map[i]
        ax = axes[ai]
        # ax = axes[ai // 3, ai % 3]
        # ax = fig.add_subplot(gs[ai // 3, ai % 3])
        ax.axis("off")  # equal
        ax.imshow(image)
        # ax.imshow(image, aspect="auto")
        # ax.set_title(f"Camera {i+1}")
        # plt.title(f'camera {i}')
        # plt.imshow(image)
        # plt.show()

    if True:
        # Initialize BEV image with zeros, here 103x100 to fit your given view range
        bev_image = np.zeros((50, 50, 3), dtype=np.uint8)

        ax = axes[6]  # [1, 2]
        # bev_ax = fig.add_subplot(gs[1, 2])

        if with_gt:
            for bbox in anno["bbox"]:
                points = get_8_corners_kitti_gt(bbox).T
                draw_bev_bbox(ax, bev_image, points, color=(0, 0, 1))

        # Assume `fbg` is your list of bounding boxes in BEV coordinates
        for bbox in fbg:
            points = get_8_corners_kitti(bbox).T
            draw_bev_bbox(ax, bev_image, points, color=(1, 0, 0))

        # Draw circles at various radii
        for r in [10, 20, 40, 60, 100]:
            draw_range_circle(ax, bev_image, r)

        # print(fbg_p.shape)
        # draw_points(ax, bev_image, a[:600, ...].reshape(-1, 3))
        draw_point_cloud(ax, bev_image, fbg_p)

        ax.axis("off")  # equal
        ax.set_xlim([0, 60])
        ax.set_ylim([-30, 30])
        ax.set_aspect("auto")
        # ax.set_title("BEV")

    # Adjust spacing between subplots to remove gaps
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # plt.tight_layout()

    # plt.show()
    # save the figure
    if with_gt:
        output_dir = f"./{save_folder_root}/plt_bbox_with_gt"
    else:
        output_dir = f"./{save_folder_root}/plt_bbox"
    if custom_suffix is not None:
        output_dir += custom_suffix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"visualize_{sample_idx}.png")
    plt.savefig(output_path)
    plt.close()
