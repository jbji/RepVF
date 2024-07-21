import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid
from ..util import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class PseudoAssigner3D(BaseAssigner):
    def __init__(self, group_num=5):
        self.group_num = group_num

    def assign(
        self, bbox_pred, cls_pred, gt_bboxes, gt_labels, gt_bboxes_ignore=None, eps=1e-7
    ):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

            # Calculate group size
        group_size = num_bboxes // self.group_num

        for i in range(self.group_num):
            start_idx = i * group_size
            end_idx = (
                start_idx + num_gts
            )  # Assign to all ground truth boxes for each group

            # Assign ground truth indices from 1 to n for each group
            assigned_gt_inds[start_idx:end_idx] = torch.arange(
                1, num_gts + 1, dtype=torch.long
            ).to(bbox_pred.device)
            assigned_labels[start_idx:end_idx] = gt_labels

            # Assign 0 (background) to the remaining predictions in the group
            assigned_gt_inds[end_idx : (i + 1) * group_size] = 0

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)


@BBOX_ASSIGNERS.register_module()
class PseudoAssignerLane3D(BaseAssigner):
    def __init__(
        self,
        group_num=5,
    ):
        self.group_num = group_num

    def assign(
        self, preds_xyz, cls_scores, gt_xyz, gt_cls, eps=1e-7  # vis_scores, gt_vis,
    ):
        num_gts, num_preds = gt_xyz.size(0), preds_xyz.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = preds_xyz.new_full((num_preds,), -1, dtype=torch.long)
        assigned_labels = preds_xyz.new_full((num_preds,), -1, dtype=torch.long)
        # assigned_vis = preds_xyz.new_full((num_preds,), -1, dtype=torch.long)
        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

            # Calculate group size
        group_size = num_preds // self.group_num

        for i in range(self.group_num):
            start_idx = i * group_size
            end_idx = (
                start_idx + num_gts
            )  # Assign to all ground truth boxes for each group

            # Assign ground truth indices from 1 to n for each group
            assigned_gt_inds[start_idx:end_idx] = torch.arange(
                1, num_gts + 1, dtype=torch.long
            ).to(preds_xyz.device)
            assigned_labels[start_idx:end_idx] = gt_cls

            # Assign 0 (background) to the remaining predictions in the group
            assigned_gt_inds[end_idx : (i + 1) * group_size] = 0

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
