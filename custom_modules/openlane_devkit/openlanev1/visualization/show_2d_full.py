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
    draw_lane,
    draw_bev_bbox,
    draw_bev_lane,
    draw_vector,
    draw_point_cloud,
    draw_range_circle,
    get_8_corners_kitti,
    get_8_corners_kitti_gt,
    project_to_image,
    project_to_image_lidar2img,
)


def preprocess_draw_vector_2d(field, vector_length, intrinsic, extrinsic):
    point_starts = field[:, :3]
    vec_d = field[:, 3:5]
    thetas = np.arctan2(vec_d[:, 0], vec_d[:, 1])
    thetas = thetas.reshape(-1, 1)
    zeros_ends = np.zeros((vec_d.shape[0], 1))
    vec_d_expanded = np.hstack((np.cos(thetas), np.sin(thetas), zeros_ends))
    points_end = point_starts + vec_d_expanded * vector_length
    points_start_image = project_to_image(field[..., :3], intrinsic, extrinsic)
    points_end_image = project_to_image(points_end, intrinsic, extrinsic)

    return points_start_image, points_end_image


def preprocess_draw_vector_2d_bbox(
    bbox,  # for scalar computation
    bbox_p,
    vector_length,
    intrinsic,
    extrinsic,
    # scalars=[0.7744, 0.6092, 0.5575],
):
    # Calculate the average center of bbox_p
    center = np.mean(bbox_p[..., :3], axis=0)
    avg_direction = np.mean(bbox_p[..., 3:5], axis=0)
    # Calculate the angle of the average direction
    theta = np.arctan2(avg_direction[0], avg_direction[1])
    # print(theta)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    # Apply the inverse rotation to the points
    rotated_points = np.dot(bbox_p[..., :2] - center[:2], rotation_matrix.T)
    points_l = np.max(rotated_points[..., 0]) - np.min(rotated_points[..., 0])
    points_w = np.max(rotated_points[..., 1]) - np.min(rotated_points[..., 1])
    points_h = np.max(bbox_p[..., 2]) - np.min(bbox_p[..., 2])
    bbox_l, bbox_w, bbox_h = bbox[3:6]
    scalars = [bbox_l / points_l, bbox_w / points_w, bbox_h / points_h]
    rescaled_points = rotated_points * np.array(scalars[:2])

    # Apply the rotation to the points
    bbox_p[..., :2] = np.dot(rescaled_points, rotation_matrix) + center[:2]
    bbox_p[..., 2] = (bbox_p[..., 2] - center[2]) * np.array(scalars[2]) + center[2]

    return preprocess_draw_vector_2d(bbox_p, vector_length, intrinsic, extrinsic)


def show_2d_full(
    sample_idx,
    cameras_path,
    intrinsics,
    extrinsics,
    fbg,
    fbg_p,
    fbs,
    flg,
    fls,
    with_gt=False,
    custom_suffix=None,
    save_folder_root="visualizations",
    show_vector=False,
    vector_length=1,
):
    # fig, axes = plt.subplots(
    #     2,
    #     3,
    #     figsize=(57.6, 22.4),
    #     gridspec_kw={"width_ratios": [1.92, 1.92, 1.92], "height_ratios": [1.28, 0.96]},
    # )  # Adjust figsize for higher resolution
    font = {"family": "serif"}
    plt.rc("font", **font)

    fig = plt.figure(
        figsize=(57.6 / 2, 22.1 / 2)
    )  # 22.4 # Adjust figsize for higher resolution
    gs = GridSpec(
        2,
        5,
        figure=fig,
        width_ratios=[1.5, 1.92, 0.94, 0.94, 1.92],
        height_ratios=[1.28, 1.28],
    )
    axes = [
        fig.add_subplot(p)
        for p in [
            gs[0, 1],
            gs[0, 2:4],
            gs[0, 4],
            gs[1, 1:3],
            gs[1, 3:],
            gs[1, 0],
        ]
    ]
    axes.append(fig.add_subplot(gs[0, 0], projection="3d"))

    if with_gt:
        json_path = (
            cameras_path[0]
            .replace("images_0", "detection3d_1000")
            .replace(".jpg", ".json")
        )
        anno = io.json_load(json_path)

    # camera i
    index_map = [1, 0, 2, 3, 4]
    camera_names = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    cmap_bbox = plt.cm.get_cmap("Wistia", 3)  # Wistia?
    cmap_lane = plt.cm.get_cmap("cool", 22)
    for i in range(5):
        # Draw the 3D bounding box on the image using the projected points
        image = cv2.imread(cameras_path[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image.shape)
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

        for lane, c in zip(flg, fls):
            color = cmap_lane(c.argmax())
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            if lane.shape[1] == 5 and show_vector:
                points_start_image, points_end_image = preprocess_draw_vector_2d(
                    lane, vector_length, intrinsic, extrinsic
                )
                if not isinstance(points_start_image, np.ndarray) or not isinstance(
                    points_end_image, np.ndarray
                ):
                    continue
                image = draw_vector(
                    image, points_start_image, points_end_image, color=color
                )
            points_image = project_to_image(lane[..., :3], intrinsic, extrinsic)
            if not isinstance(points_image, np.ndarray):
                continue
            image = draw_lane(image, points_image, color=color)

        for bbox, bbox_p, c in zip(fbg, fbg_p, fbs):
            color = cmap_bbox(2 - c.argmax())
            color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            # bbox = annotation['bbox'][15]

            points_image = project_to_image(
                get_8_corners_kitti(bbox).T, intrinsic, extrinsic
            )
            if not isinstance(points_image, np.ndarray):
                continue
            # print(points_image)
            image = draw_bbox(image, points_image, color=color)
            # break

            if bbox_p.shape[1] == 5 and show_vector:

                points_start_image, points_end_image = preprocess_draw_vector_2d_bbox(
                    bbox, bbox_p, vector_length, intrinsic, extrinsic
                )
                if not isinstance(points_start_image, np.ndarray) or not isinstance(
                    points_end_image, np.ndarray
                ):
                    continue

                image = draw_vector(
                    image, points_start_image, points_end_image, color=color
                )

        ai = index_map[i]
        ax = axes[ai]
        # ax = axes[ai // 3, ai % 3]
        # ax = fig.add_subplot(gs[ai // 3, ai % 3])
        ax.axis("off")  # equal
        ax.imshow(image)
        # ax.imshow(image, aspect="auto")
        ax.set_title(f"CAMERA_{camera_names[i]}", fontsize=16)
        # plt.title(f'camera {i}')
        # plt.imshow(image)
        # plt.show()

    if True:
        # Initialize BEV image with zeros, here 103x100 to fit your given view range
        bev_image = np.zeros((50, 50, 3), dtype=np.uint8)

        ax = axes[5]  # [1, 2]
        # bev_ax = fig.add_subplot(gs[1, 2])

        if with_gt:
            for bbox in anno["bbox"]:
                points = get_8_corners_kitti_gt(bbox).T
                x, y = points[:, 0].copy(), points[:, 1].copy()
                points[:, 0], points[:, 1] = -y, x
                draw_bev_bbox(ax, bev_image, points, color=(0, 0, 1))

        # Draw circles at various radii
        for r in [10, 20, 40, 80]:
            draw_range_circle(ax, bev_image, r)

        for lane, c in zip(flg, fls):
            points = lane.copy()
            x, y = points[:, 0].copy(), points[:, 1].copy()
            points[:, 0], points[:, 1] = -y, x
            draw_bev_lane(ax, bev_image, points, color=cmap_lane(c.argmax()))

        # Assume `fbg` is your list of bounding boxes in BEV coordinates
        for bbox, c in zip(fbg, fbs):
            points = get_8_corners_kitti(bbox).T
            x, y = points[:, 0].copy(), points[:, 1].copy()
            points[:, 0], points[:, 1] = -y, x
            draw_bev_bbox(ax, bev_image, points, color=cmap_bbox(2 - c.argmax()))

        # print(fbg_p.shape)
        # draw_points(ax, bev_image, a[:600, ...].reshape(-1, 3))
        # draw_point_cloud(ax, bev_image, fbg_p)

        ax.axis("off")  # equal
        ax.set_ylim([0, 90])
        ax.set_xlim([-45, 45])
        # ax.set_aspect("auto")
        # ax.set_title("TOP_VIEW", fontsize=16)

    if True:
        ax = axes[6]
        # for bbox, c in zip(fbg, fbs):
        #     points = get_8_corners_kitti(bbox).T
        #     ax.plot(points[0, :], points[1, :], color=cmap_bbox(c.argmax()))

        for lane, c in zip(flg, fls):
            ax.plot(-lane[:, 1], lane[:, 0], color=cmap_lane(c.argmax()), linewidth=3)

        # ax.set_aspect("equal")
        ax.set_box_aspect(None)
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_zlabel("Z")
        ax.zaxis.labelpad = 2  # Adjust the padding for the Z label if needed.
        # control the plot range
        ax.set_xlim(-20, 20)
        ax.set_ylim(3, 88)
        ax.set_zlim(-10, 10)

    # Adjust spacing between subplots to remove gaps
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    plt.tight_layout()

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
    output_path = os.path.join(output_dir, f"visualize_{sample_idx}.pdf")
    plt.savefig(output_path, format="pdf")
    plt.close()
