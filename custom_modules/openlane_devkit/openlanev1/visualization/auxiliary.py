# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

import os

import cv2
import matplotlib.patches as patches
import numpy as np

import matplotlib.pyplot as plt
from ipywidgets import interact, widgets


def draw_bbox(image, points, color=(0, 0, 255)):
    """
    This function takes an image and a list of points
    then it draws the bounding box on the image and returns it
    """
    # Connect each point to its adjacent ones
    points = points.astype(int)
    for i in range(4):
        cv2.circle(image, tuple(points[i]), 5, color, -1)
        cv2.circle(image, tuple(points[i + 4]), 5, color, -1)
        line_width = 4
        cv2.line(
            image, tuple(points[i]), tuple(points[(i + 1) % 4]), color, line_width
        )  # bottom square
        cv2.line(
            image,
            tuple(points[i + 4]),
            tuple(points[(i + 1) % 4 + 4]),
            color,
            line_width,
        )  # top square
        cv2.line(
            image, tuple(points[i]), tuple(points[i + 4]), color, line_width
        )  # vertical lines

    # Create an overlay of the same size as the original image
    overlay = image.copy()
    front_face = [points[0], points[3], points[7], points[4]]  # bottom front face
    cv2.fillPoly(overlay, [np.array(front_face)], color)

    # Combine the overlay and the original image
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

    return image


def draw_lane(image, points, color=(0, 0, 255)):
    """
    This function takes an image and a list of points
    then it draws the bounding box on the image and returns it
    """
    # Connect each point to its adjacent ones
    # print(points)
    points = points.astype(int)
    # print(points)
    # Connect each point to its adjacent ones
    for i in range(len(points) - 1):
        point1 = tuple(points[i])
        point2 = tuple(points[i + 1])
        cv2.line(image, point1, point2, color, 5)

    return image


def draw_vector(image, points_start_image, points_end_image, color):
    for start, end in zip(points_start_image, points_end_image):
        start = tuple(start.astype(int))
        end = tuple(end.astype(int))
        image = cv2.arrowedLine(image, start, end, color, thickness=6, tipLength=0.3)
    return image


def project_to_image(pts_3d, intrinsic, extrinsic):
    E_inv = np.linalg.inv(extrinsic)[0:3, :]
    P_g2im = np.matmul(intrinsic, E_inv)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=-1)
    pts_2d = (P_g2im @ pts_3d_homo.T)[:3, :]
    # bbox not in the frustrum space would be in the back of the camera.
    if np.any(pts_2d[2, :] <= 0):
        return False
    pts_2d = pts_2d[:2, :] / pts_2d[2, :]
    return pts_2d.T


def project_to_image_lidar2img(pts_3d, lidar2img):
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=-1)
    pts_2d = (lidar2img @ pts_3d_homo.T)[:3, :]
    # bbox not in the frustrum space would be in the back of the camera.
    if np.any(pts_2d[2, :] <= 0):
        return False
    pts_2d = pts_2d[:2, :] / pts_2d[2, :]
    return pts_2d.T


# def draw_bev_bbox(plt, bev_image, points, color=(1, 0, 0), front_face_color=(0, 1, 0)):
#     points = points.astype(int)
#     for i in range(4):
#         plt.plot(
#             [points[i, 0], points[(i + 1) % 4, 0]],
#             [points[i, 1], points[(i + 1) % 4, 1]],
#             color=color,
#         )

#     # Draw front face with a different color
#     plt.plot(
#         [points[3, 0], points[0, 0]],
#         [points[3, 1], points[0, 1]],
#         color=front_face_color,
#     )


def draw_bev_bbox(plt, bev_image, points, color=(1, 0, 0), front_face_color=(0, 1, 0)):
    # points = points.astype(int)
    plt.plot([points[0, 0], points[1, 0]], [points[0, 1], points[1, 1]], color=color)
    plt.plot([points[1, 0], points[2, 0]], [points[1, 1], points[2, 1]], color=color)
    plt.plot([points[2, 0], points[3, 0]], [points[2, 1], points[3, 1]], color=color)
    # Draw front face with a different color
    plt.plot(
        [points[3, 0], points[0, 0]],
        [points[3, 1], points[0, 1]],
        color=front_face_color,
    )


def draw_bev_lane(plt, bev_image, points, color=(1, 0, 0)):
    # points = points.astype(int)
    # Draw the lane on the BEV image
    for i in range(len(points) - 1):
        point1 = tuple(points[i])
        point2 = tuple(points[i + 1])
        plt.plot(
            [point1[0], point2[0]], [point1[1], point2[1]], color=color, linewidth=2
        )


def draw_points(plt, bev_image, pcd, color=(0, 0, 0)):
    x = pcd[:, 0]  # .astype(int)
    y = pcd[:, 1]  # .astype(int)
    plt.scatter(x, y, c=color, s=0.5)


def draw_point_cloud(plt, bev_image, point_cloud, color="g"):
    for query_points in point_cloud:
        x = query_points[:, 0]  # .astype(int)
        y = query_points[:, 1]  # .astype(int)
        plt.scatter(x, y, c=color, s=1)

        x = query_points[0, 0]  # .astype(int)
        y = query_points[0, 1]  # .astype(int)
        plt.scatter(x, y, c="b", s=1)


def draw_range_circle(plt, bev_image, radius, color="b"):
    circle = patches.Circle((0, 0), radius, fill=False, color=color)
    plt.add_patch(circle)


def get_8_corners_kitti_gt(bbox: dict):
    """
    Note: the bbox is in kitti format, so is the point cloud
    The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)
    """
    h, w, l = bbox["dimensions"]
    x, y, z = bbox["center_location"]
    rz = bbox["rotation_y"]  # the name is inaccruate cause y and z are swapped in kitti

    corners = np.array(
        [
            [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2],
            [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2],
            [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
        ]
    )

    # rotation must be applied before translation, this is because:
    # the rotation matrix is in the world coordinate system
    rotation_matrix = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )

    corners = np.dot(rotation_matrix, corners)

    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def get_8_corners_kitti(bbox):
    """
    Note: the bbox is in kitti format, so is the point cloud
    The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)
    """
    l, w, h = bbox[3:6]
    x, y, z = bbox[:3]
    rz = bbox[6]  # the name is inaccruate cause y and z are swapped in kitti

    corners = np.array(
        [
            [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2],
            [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2],
            [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
        ]
    )

    # rotation must be applied before translation, this is because:
    # the rotation matrix is in the world coordinate system
    rotation_matrix = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )

    corners = np.dot(rotation_matrix, corners)

    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners
