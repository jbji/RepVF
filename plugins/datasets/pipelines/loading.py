# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np

import mmcv
from mmdet.datasets import PIPELINES

from mmdet.datasets.pipelines import LoadImageFromFile

from openlanev1.utils import npdump
import os
import shutil


@PIPELINES.register_module()
class CustomLoadImageFromFile(LoadImageFromFile):
    """
    Load image from file and convert it to a numpy array.
    to_float32 (bool): Whether to convert the image to a float32 numpy array.

    Add: img, img_shape
    Update: None
    """

    def __call__(self, results):
        filenames = results["img_path"]
        imgs = []
        for filename in filenames:
            img = mmcv.imread(filename, self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)

        results["img"] = imgs
        results["img_shape"] = [img.shape for img in imgs]
        return results


pseudo_idx_counter = {"images": 0}


@PIPELINES.register_module()
class CustomLoadImagesFromFile(LoadImageFromFile):
    def __call__(self, results):
        filenames = results["img_paths"]
        imgs = []
        for filename in filenames:
            img = mmcv.imread(filename, self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)

        results["img"] = imgs
        results["img_shape"] = [img.shape for img in imgs]

        if os.environ.get("SAVE_FOR_VISUALIZATION") == "True":
            global pseudo_idx_counter
            dump_limit = int(os.environ.get("DUMP_LIMIT", 1000))  # Default limit
            dump_step = int(os.environ.get("DUMP_STEP", 5))  # Default step
            pseudo_idx = pseudo_idx_counter["images"]
            if pseudo_idx % dump_step == 0 and pseudo_idx < dump_limit:
                sample_idx = results["sample_idx"]
                os.makedirs(f"./visualizations/images/{sample_idx}", exist_ok=True)
                for i, filename in enumerate(filenames):
                    shutil.copy(
                        filename, f"./visualizations/images/{sample_idx}/{i}.jpg"
                    )
            pseudo_idx_counter["images"] += 1

        return results


@PIPELINES.register_module()
class LoadAnnotations3DLane:
    def __init__(
        self,
        camera_frustum_to_world=True,
        apollo_to_openlane=True,
        merge_road_edge=True,
        **kwargs,
    ):
        self.apollo_to_openlane = apollo_to_openlane
        self.camera_frustum_to_world = camera_frustum_to_world
        self.merge_road_edge = merge_road_edge

    def __call__(self, results):
        gt_lanes_packed = results["ann_info"]["gt_lane_lines"]
        self.lane_count = len(gt_lanes_packed)

        gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []

        for i, gt_lane_packed in enumerate(gt_lanes_packed):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = gt_lane_packed["xyz"]
            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))

            if self.camera_frustum_to_world:
                # lane labels are defined on front camera.
                lane = np.matmul(results["calib"]["extrinsics"][0], lane)

            if self.apollo_to_openlane:
                cam_representation = np.linalg.inv(
                    np.array(
                        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                        dtype=float,
                    )
                )  # transformation from apollo camera to openlane camera
                lane = np.matmul(
                    results["calib"]["extrinsic"], np.matmul(cam_representation, lane)
                )

            lane = lane[0:3, :].T
            # lane = lane.astype(np.float32)
            gt_lane_pts.append(lane)

            gt_lane_visibility.append(gt_lane_packed["visibility"])

            if "category" in gt_lane_packed:
                lane_c = gt_lane_packed["category"]
                if (
                    self.merge_road_edge and lane_c == 21
                ):  # merge left and right road edge into road edge
                    lane_c = 20
                gt_laneline_category.append(lane_c)
            else:
                gt_laneline_category.append(1)

        results.update(
            {
                "gt_lanes_3d": {
                    "lane_pts": gt_lane_pts,
                    "lane_visibility": gt_lane_visibility,
                    "lane_category": gt_laneline_category,
                }
            }
        )
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"apollo_to_openlane={self.apollo_to_openlane}, "
            f"merge_road_edge={self.merge_road_edge},"
            f")"
        )
        return repr_str


@PIPELINES.register_module()
class LoadLaneAnnotationsFromResults:
    """
    Load lane annotations into the format of openlane_300, runtime conversion

    LaneShape had changed from (3,N) to (N,3).

    Note: I don't know if two copies of annotations would cause a potential performance issue.

    Add: gt_lanes : dict of lane_pts, lane_visibility, laneline_category, each one is a list of numpy arrays
    """

    def __init__(self, apollo_to_openlane=True, merge_road_edge=True, **kwargs):
        self.apollo_to_openlane = apollo_to_openlane
        self.merge_road_edge = merge_road_edge

    def __call__(self, results):
        gt_lanes_packed = results["gt_lane_lines"]
        self.lane_count = len(gt_lanes_packed)

        gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []

        for i, gt_lane_packed in enumerate(gt_lanes_packed):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = gt_lane_packed["xyz"]

            if self.apollo_to_openlane:
                # Coordinate convertion for openlane_300 data
                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                cam_representation = np.linalg.inv(
                    np.array(
                        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                        dtype=float,
                    )
                )  # transformation from apollo camera to openlane camera
                lane = np.matmul(
                    results["calib"]["extrinsic"], np.matmul(cam_representation, lane)
                )

            lane = lane[0:3, :].T
            gt_lane_pts.append(lane)

            gt_lane_visibility.append(gt_lane_packed["visibility"])

            if "category" in gt_lane_packed:
                lane_c = gt_lane_packed["category"]
                if (
                    self.merge_road_edge and lane_c == 21
                ):  # merge left and right road edge into road edge
                    lane_c = 20
                gt_laneline_category.append(lane_c)
            else:
                gt_laneline_category.append(1)

        results.update(
            {
                "gt_lanes": {
                    "lane_pts": gt_lane_pts,
                    "lane_visibility": gt_lane_visibility,
                    "lane_category": gt_laneline_category,
                }
            }
        )
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"apollo_to_openlane={self.apollo_to_openlane}, "
            f"merge_road_edge={self.merge_road_edge},"
            f")"
        )
        return repr_str
