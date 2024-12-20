# Modified from PETR (https://github.com/megvii-research/PETR)
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from .dgcnn3d_head import DGCNN3DHead
from .detr3d_head import Detr3DHead
from .petr_head import PETRHead
from .rftr_head import RFTRHead

__all__ = [
    "DGCNN3DHead",
    "Detr3DHead",
    "PETRHead",
    "RFTRHead",
]
