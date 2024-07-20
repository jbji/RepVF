from os.path import exists, expanduser, join

import cv2
import numpy as np
from openlanev1.io import io


class MultiViewFrame:
    r"""
    A data structure containing meta data of a frame.

    """

    def __init__(self, root_path: str, meta: dict) -> None:
        r"""
        Parameters
        ----------
        root_path : str
        meta : dict
            Meta data of a frame.

        """
        self.root_path = expanduser(root_path)
        self.meta = meta

    def get_point_cloud_path(self) -> str:
        return join(self.root_path, self.meta["path_point_cloud"])

    def get_image_paths(self) -> list:
        return [join(self.root_path, path) for path in self.meta["path_cameras"]]

    def get_rgb_images(self) -> np.ndarray:
        r"""
        Returns the RGB image of the current frame.

        Parameters
        ----------
        camera : str

        Returns
        -------
        np.ndarray
            RGB Image.

        """
        image_paths = self.get_image_paths()
        return [
            cv2.cvtColor(io.cv2_imread(image_path), cv2.COLOR_BGR2RGB)
            for image_path in image_paths
        ]

    def get_intrinsics(self) -> dict:  # only one camera
        return self.meta["intrinsics"]

    def get_extrinsics(self) -> dict:
        return self.meta["extrinsics"]

    def get_annotations(self) -> dict:
        return {
            "lane_lines": self.meta["lane_lines"],
            "gt_boxes": self.meta["gt_boxes"],
            "gt_names": self.meta["gt_names"],
            "gt_velocity": self.meta["gt_velocity"],
        }

    def get_annotations_bbox3d_velocity(self) -> list:
        return self.meta["gt_velocity"]

    def get_annotations_bbox3d_names(self) -> list:
        return self.meta["gt_names"]

    def get_annotations_bbox3d(self) -> list:
        return self.meta["gt_boxes"]

    def get_annotations_lane_lines(self) -> list:
        return self.meta["lane_lines"]
