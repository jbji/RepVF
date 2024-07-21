# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.


import numpy as np
from scipy.interpolate import interp1d

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
import mmcv
from mmdet.datasets import PIPELINES
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
import torch
from openlanev1.utils import npdump

# update camera parameters
# formatting annotation, runtime conversion


@PIPELINES.register_module()
class UpdateCameraParameters:
    """
    Re-calculate extrinsic matrix based on ground coordinate

    Add: gt_cam_height, gt_cam_pitch
    Update:  calib - extrinsic, calib - intrinsic
    """

    def __init__(self, for_unify=True):
        """update camera parameters

        Args:
            for_bbox (bool, optional): check this if you work on both openlane and waymo, which aligns all camera(including waymos) to kitti style. Defaults to False, which assumes only work on openlane dataset which has only one camera, and all cameras are aligned to 3DLaneNet way.
        """
        self.for_unify, self.for_lane = for_unify, not for_unify

    def __call__(self, results):
        # in case only working with openlanev1 with no waymo addon

        if self.for_unify and "extrinsics" in results["calib"]:
            intrinsics = []
            extrinsics = []
            assert (
                len(results["calib"]["extrinsics"]) == 5
            ), "Number of cameras is not 5"
            for i in range(5):
                extrinsic = np.vstack([results["calib"]["extrinsics"][i], [0, 0, 0, 1]])
                intrinsic = np.array(results["calib"]["intrinsics"][i])

                # R_x = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
                # R_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                waymo_to_stdcam = np.array(
                    [[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float
                )
                # intrinsics are for stdcam, i.e. z front, x right, y down
                # but extrinsics are for waymo cam, i.e. x front, y left, z up
                # two fixes, either align the extrinsics to the intrinsics, or align the intrinsics to the extrinsics
                # since lane label and bbox label are both in coordinate x front y left z up, we align the intrinsics to the extrinsics
                # this fixes the misalignment
                intrinsic = intrinsic @ waymo_to_stdcam
                # extrinsic[:3, :3] = extrinsic[:3, :3] @ waymo_to_std

                extrinsics.append(extrinsic)
                intrinsics.append(intrinsic)

            results["calib"].update(
                {"extrinsics": extrinsics, "intrinsics": intrinsics}
            )
            results["calib_original"] = {
                "extrinsics": [np.copy(e) for e in extrinsics],
                "intrinsics": [np.copy(i) for i in intrinsics],
            }

        # this is for lane det only
        if self.for_lane:
            cam_extrinsic = results["calib"]["extrinsic"]["rot_trans"]
            # NOTE: camera-center based axis(gt label) to ground-based axis()
            # Re-calculate extrinsic matrix based on ground coordinate
            # R_vg and R_gc stands for rotation vehicle-ground and rotation ground-camera？
            # camera: y: height z: depth
            # ground: y: forward, z: height
            R_vg = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
            R_gc = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
            # by testing cam_extrinsics[:3,:3].T@cam_extrinsics[:3,:3] ~ I, (R.T @ R = I)
            # and np.linalg.det(cam_extrinsics[:3,:3]) = 1.0
            # we can say that cam_extrinsics[:3,:3] is solely a rotation matrix
            # this is completely unnecessay

            cam_extrinsic[:3, :3] = np.matmul(
                np.matmul(np.matmul(np.linalg.inv(R_vg), cam_extrinsic[:3, :3]), R_vg),
                R_gc,
            )
            #
            # defining ground origin at the camera center
            cam_extrinsic[0:2, 3] = 0.0

            results["calib"].update(
                {
                    "extrinsic": cam_extrinsic,
                    "intrinsic": results["calib"]["intrinsic"]["k"],
                }
            )

        cam_extrinsic = extrinsics[0]
        results.update(
            {
                "gt_cam_height": cam_extrinsic[2, 3],  # camera -> ground height?
                # waymo could be defining rear part of car as lidar(x) ground origin
                "gt_cam_pitch": 0,
            }
        )

        npdump(
            out_type="calibs",
            np_obj=np.array(results["calib"]),
            img_metas=[{"sample_idx": results["sample_idx"]}],
        )

        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f")"
        return repr_str


@PIPELINES.register_module()
class Lane3DRangeVisFilter:
    def __init__(
        self,
        by_visibility=True,
        visibility_threshold=0,
        by_range=True,
        y_range=(-10, 10, 3),
        x_range_to_prune=(0, 200),
    ):
        self.by_visibility = by_visibility
        self.visibility_threshold = visibility_threshold
        self.by_range = by_range
        self.y_min, self.y_max, self.y_scale = y_range
        self.x_min, self.x_max = x_range_to_prune

    def filter_by_visibility(self, lanes, visibility):
        return [
            lane[visibility[idx] > self.visibility_threshold]
            for idx, lane in enumerate(lanes)
        ]

    def filter_by_range(self, lanes):
        return [self.prune_3d_lane_by_range(lane) for lane in lanes]

    def prune_3d_lane_by_range(self, lane):
        within_y = np.logical_and(
            lane[:, 1] > self.y_min * self.y_scale,
            lane[:, 1] < self.y_max * self.y_scale,
        )
        within_x = np.logical_and(lane[:, 0] > self.x_min, lane[:, 0] < self.x_max)
        return lane[np.logical_and(within_y, within_x), ...]

    def filter_empty_lanes(self, lanes, visibility, category):
        non_empty_idxs = [idx for idx, lane in enumerate(lanes) if lane.shape[0] > 1]
        return (
            [lanes[idx] for idx in non_empty_idxs],
            [visibility[idx] for idx in non_empty_idxs],
            [category[idx] for idx in non_empty_idxs],
        )

    def __call__(self, results):
        lanes = results["gt_lanes_3d"]["lane_pts"]
        visibility = results["gt_lanes_3d"]["lane_visibility"]
        category = results["gt_lanes_3d"]["lane_category"]

        if self.by_visibility:
            lanes = self.filter_by_visibility(lanes, visibility)

        if self.by_range:
            lanes = self.filter_by_range(lanes)

        lanes, visibility, category = self.filter_empty_lanes(
            lanes, visibility, category
        )

        results["gt_lanes_3d"] = {
            "lane_pts": lanes,
            "lane_visibility": visibility,
            "lane_category": category,
        }
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@PIPELINES.register_module()
class Lane3DResampler:
    def __init__(self, num_points=10):
        self.num_points = num_points

    def resample_lane(self, lane, visibility):
        """Resample the lane and visibility to have num_points points."""
        # print(lane.shape, visibility.shape)
        x = lane[:, 0]
        y = lane[:, 1]
        z = lane[:, 2]

        # Create an interpolating function for y, z, and visibility based on x
        f_y = interp1d(x, y, kind="linear", fill_value="extrapolate")
        f_z = interp1d(x, z, kind="linear", fill_value="extrapolate")
        f_visibility = interp1d(
            x, visibility[: x.shape[0]], kind="nearest", fill_value="extrapolate"
        )

        # Generate new x values for resampling
        x_new = np.linspace(min(x), max(x), self.num_points)
        y_new = f_y(x_new)
        z_new = f_z(x_new)
        visibility_new = f_visibility(x_new)

        # Combine the resampled points
        lane_resampled = np.column_stack((x_new, y_new, z_new))

        return lane_resampled, visibility_new

    def __call__(self, results):
        lanes = results["gt_lanes_3d"]["lane_pts"]
        visibility = results["gt_lanes_3d"]["lane_visibility"]

        # Resample each lane
        for i in range(len(lanes)):
            lanes[i], visibility[i] = self.resample_lane(lanes[i], visibility[i])

        # Update results
        results["gt_lanes_3d"]["lane_pts"] = lanes
        results["gt_lanes_3d"]["lane_visibility"] = visibility

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(num_points={self.num_points})"


@PIPELINES.register_module()
class CustomFormatBundle3D(DefaultFormatBundle3D):
    """This formats 3dlane for extra

    - gt_lanes_3d: (1) to_tensor, (2) to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super().__init__(class_names, with_gt, with_label)

    def format_class_scores(self, class_scores):
        """Converts class_scores to a single NumPy array."""
        return np.array(class_scores, dtype=np.int64)

    def format_xyz_visibility(self, lanes, visibility):
        """Converts lanes and visibility, which are lists of arrays,
        to single NumPy arrays with one extra dimension.
        """
        if not lanes:
            return np.empty((0, 10, 3), dtype=np.float32), np.empty(
                (10, 0), dtype=np.float32
            )
        lanes_array = np.stack(lanes, axis=0).astype(np.float32)
        visibility_array = np.stack(visibility, axis=0).astype(np.float32)
        return lanes_array, visibility_array

    def __call__(self, results):
        if self.with_label:
            class_scores = results["gt_lanes_3d"]["lane_category"]
            lanes = results["gt_lanes_3d"]["lane_pts"]
            visibility = results["gt_lanes_3d"]["lane_visibility"]

            # Format class_scores
            results["gt_lanes_3d"]["lane_category"] = DC(
                to_tensor(self.format_class_scores(class_scores))
            )

            # Format lanes and visibility
            lanes_array, visibility_array = self.format_xyz_visibility(
                lanes, visibility
            )
            results["gt_lanes_3d"]["lane_pts"] = DC(to_tensor(lanes_array))
            results["gt_lanes_3d"]["lane_visibility"] = DC(to_tensor(visibility_array))

        results = super().__call__(results)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@PIPELINES.register_module()
class Prune3dLaneAnnotations:
    """
    Prune 3d lane points based on visibility and range,
    Coordinate System: y front, x left, z up
    Add: None
    Update: gt_lanes - lane_pts, lane_visibility, lane_category
    i'm wh,my son
    """

    def __init__(
        self,
        by_visibility=True,
        visibility_threshold=0,
        by_range=True,
        x_range=(-10, 10, 3),
        prune_y_range=(0, 200),
    ):
        self.by_visibility = by_visibility
        self.visibility_threshold = visibility_threshold
        self.by_range = by_range
        self.x_min, self.x_max, self.x_scale = x_range
        self.y_min, self.y_max = prune_y_range

    def prune_3d_lane_by_range(self, lane_3d, x_min, x_max, y_min, y_max):
        # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
        # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
        lane_3d = lane_3d[
            np.logical_and(lane_3d[:, 1] > y_min, lane_3d[:, 1] < y_max), ...
        ]

        # remove lane points out of x range
        lane_3d = lane_3d[
            np.logical_and(lane_3d[:, 0] > x_min, lane_3d[:, 0] < x_max), ...
        ]
        return lane_3d

    def __call__(self, results):
        gt_lanes = results["gt_lanes"]["lane_pts"]
        gt_visibility = results["gt_lanes"]["lane_visibility"]
        gt_category = results["gt_lanes"]["lane_category"]

        if self.by_visibility:
            # prune gt lanes by visibility labels
            gt_lanes = [
                gt_lane[gt_visibility[k] > self.visibility_threshold, ...]
                for k, gt_lane in enumerate(gt_lanes)
            ]

        if self.by_range:
            # prune out-of-range points are necessary before transformation
            gt_lanes = [
                self.prune_3d_lane_by_range(
                    gt_lane,
                    self.x_scale * self.x_min,
                    self.x_scale * self.x_max,
                    self.y_min,
                    self.y_max,
                )
                for gt_lane in gt_lanes
            ]

        # prune null lanes (this is because that some lanes coule be empty after pruning)

        # firstly also prune visibility and category
        gt_visibility = [
            gt_visibility[k]
            for k, gt_lane in enumerate(gt_lanes)
            if gt_lane.shape[0] > 1
        ]
        gt_category = [
            gt_category[k] for k, gt_lane in enumerate(gt_lanes) if gt_lane.shape[0] > 1
        ]

        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # prune null lanes
        results.update(
            {
                "gt_lanes": {
                    "lane_pts": gt_lanes,
                    "lane_visibility": gt_visibility,
                    "lane_category": gt_category,
                }
            }
        )

        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f")"
        return repr_str
