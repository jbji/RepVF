# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# frame.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import cv2
import numpy as np

from ..io import io
from os.path import join, expanduser

class Frame:
    r"""
    A data structure containing meta data of a frame.

    """
    def __init__(self, root_path : str, meta : dict) -> None:
        r"""
        Parameters
        ----------
        root_path : str
        meta : dict
            Meta data of a frame.

        """
        self.root_path = expanduser(root_path)
        self.meta = meta

    def get_image_path(self) -> str:
        r"""
        Retuens the image path of the current frame.

        Parameters
        ----------
        camera : str

        Returns
        -------
        str
            Image path.

        """
        return join(self.root_path, self.meta['file_path'])

    def get_rgb_image(self) -> np.ndarray:
        r"""
        Retuens the RGB image of the current frame.

        Parameters
        ----------
        camera : str

        Returns
        -------
        np.ndarray
            RGB Image.

        """
        image_path = self.get_image_path()
        return cv2.cvtColor(io.cv2_imread(image_path), cv2.COLOR_BGR2RGB)

    def get_intrinsic(self) -> dict: # only one camera
        r"""
        Retuens the intrinsic of the current frame.

        Parameters
        ----------
        camera : str

        Returns
        -------
        dict
            {'K': [3, 3], 'distortion': [3, ]}.

        """
        return self.meta['intrinsic']

    def get_extrinsic(self) -> dict:
        r"""
        Retuens the extrinsic of the current frame.

        Parameters
        ----------
        camera : str

        Returns
        -------
        dict
            {'rotation': [3, 3], 'translation': [3, ]}.

        """
        return self.meta['extrinsic']

    def get_annotations(self) -> dict:
        r"""
        Retuens annotations of the current frame.

        Returns
        -------
        dict
            {'lane_lines': list of lane lines, each one is a dict {category, visibility, uv, xyz} }.

        """
        if 'lane_lines' not in self.meta:
            return None
        else:
            return {'lane_lines': self.meta['lane_lines']}

    def get_annotations_lane_lines(self) -> list:
        r"""
        Retuens lane line annotations of the current frame.

        Returns
        -------
        list
            ist of dict {class, visibility, uv, xyz}
        """
        result = self.get_annotations()
        return result['lane_lines'] if result is not None else result


