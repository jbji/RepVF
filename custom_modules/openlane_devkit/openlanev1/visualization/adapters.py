# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_predictions_and_geometry(semantic_array, geometry_array, k, threshold=0.5):
    array_sigmoid = sigmoid(semantic_array)  # Apply sigmoid
    aggregate_scores = np.max(array_sigmoid, axis=-1)
    sorted_indices = np.argsort(-aggregate_scores, axis=-1)  # Sort indices
    top_k_indices = sorted_indices[..., :k]  # Take top-k

    filtered_semantics = np.take_along_axis(
        array_sigmoid, np.expand_dims(top_k_indices, axis=-1), axis=-2
    )
    filtered_geometries = np.take_along_axis(
        geometry_array, np.expand_dims(top_k_indices, axis=-1), axis=-2
    )
    filtered_aggregate_semantics = np.max(filtered_semantics, axis=-1)
    threshold_mask = (
        filtered_aggregate_semantics >= threshold
    )  # Create mask based on threshold

    filtered_semantics = filtered_semantics[threshold_mask]
    filtered_geometries = filtered_geometries[threshold_mask]
    return filtered_semantics, filtered_geometries


def denormalize_bbox(normalized_bboxes, pc_range=None):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = np.arctan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = np.exp(w)
    l = np.exp(l)
    h = np.exp(h)
    denormalized_bboxes = np.concatenate([cx, cy, cz, w, l, h, rot], axis=-1)
    return denormalized_bboxes


def limit_yaw(yaw, offset=0.5, period=np.pi * 2):
    """
    Limit yaw angle within a specific range.

    Parameters:
    yaw (float): The yaw angle in radians.
    offset (float): The offset to adjust the start of the period.
    period (float): The period within which to limit the yaw. Default is 2π.

    Returns:
    float: The limited yaw angle.
    """
    return (yaw - offset) % period + offset


def adapt_results(
    a,
    bg,
    lg,
    bs,
    ls,
    ks=[300, 15],
    query_nums=[600, 150],
    bbox3d_threshold=0.5,
    lane3d_threshold=0.5,
):
    """adpat model prediction to a format that is friendly for visualization

    Args:
            a (nparray): all_outs_geometry, at specific decoder layer and batch size
            bg (nparray): bbox_geometry
            lg (nparray): lane_geometry
            bs (np array): bbox semantics
            ls (np array): lane semantics
            ks (list, optional): top_k count. Defaults to [300,15].
            query_nums (list, optional): query numbers for two tasks. Defaults to [600,150].
            bbox3d_threshold (float, optional): bbox3d filter threshold. Defaults to 0.5.
            lane3d_threshold (float, optional): lane3d filter threshold. Defaults to 0.5.
    """
    bg = denormalize_bbox(bg)
    bg[:, 6] = limit_yaw(bg[:, 6])

    fbs, fbg = filter_predictions_and_geometry(
        bs, bg, k=ks[0], threshold=bbox3d_threshold
    )
    _, fbg_p = filter_predictions_and_geometry(
        bs,
        a[: query_nums[0], ...].reshape(query_nums[0], -1),
        k=ks[0],
        threshold=bbox3d_threshold,
    )
    fbg_p = fbg_p.reshape(-1, *a.shape[-2:])
    fls, flg = filter_predictions_and_geometry(
        ls, lg.reshape(query_nums[1], -1), k=ks[1], threshold=lane3d_threshold
    )
    flg = flg.reshape(-1, *lg.shape[-2:])
    return fbs, fbg, fbg_p, fls, flg
