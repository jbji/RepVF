import torch
import torch.nn.functional as F
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.iou_calculators import bbox_overlaps


@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class Lane3DL1Cost(object):
    """Lane3DL1Cost.
    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, lane_pred, gt_lanes):
        """
        Args:
            lane_pred (Tensor): Predicted 3D lane points. Shape [num_query, N * 3].
            gt_lanes (Tensor): Ground truth 3D lane points. Shape [num_gt, N * 3].
        Returns:
            torch.Tensor: lane_cost value with weight
        """
        # Compute the cost here. You may need to modify the shape or
        # calculation to fit your specific needs.
        lane_cost = torch.cdist(lane_pred, gt_lanes, p=1)
        return lane_cost * self.weight


@MATCH_COST.register_module()
class ChamferCost(object):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, lane_pred, gt_lanes):
        """
        Args:
            lane_pred (Tensor): Predicted 3D lane points. Shape [num_query, N * 3].
            gt_lanes (Tensor): Ground truth 3D lane points. Shape [num_gt, N * 3].
        Returns:
            torch.Tensor: lane_cost value with weight
        """
        num_query, num_gt = lane_pred.size(0), gt_lanes.size(0)
        lane_pred = lane_pred.reshape(num_query, -1, 3)
        gt_lanes = gt_lanes.reshape(num_gt, -1, 3)
        lane_cost = chamfer_distance_cost_matrix(lane_pred, gt_lanes)
        # lane_cost = lane_cost.sum()
        return lane_cost * self.weight


def chamfer_distance_cost_matrix(pred, gt):
    """
    Calculates Chamfer Distance cost matrix between predictions and ground truth for Hungarian matching.

    Args:
    - pred: Predicted points, tensor of shape (B, N, 3), where B is batch size, N is number of points.
    - gt: Ground truth points, tensor of shape (M, N, 3), where M is the number of GT items, N is number of points.

    Returns:
    - cost_matrix: Chamfer Distance cost matrix of shape (B, M).
    """
    B, N, _ = pred.shape
    M, _, _ = gt.shape

    # Expand dims to compute pairwise squared distances
    pred_expanded = pred.unsqueeze(1).unsqueeze(3)  # Shape (B, 1, N, 1, 3)
    gt_expanded = gt.unsqueeze(0).unsqueeze(2)  # Shape (1, M, 1, N, 3)

    # Compute squared distances
    dist_squared = torch.sum(
        (pred_expanded - gt_expanded) ** 2, dim=-1
    )  # Shape (B, M, N)

    # Compute minimum distances from pred to gt and gt to pred
    min_dist_pred_to_gt, _ = torch.min(dist_squared, dim=3)  # Shape (B, M, N)
    min_dist_gt_to_pred, _ = torch.min(
        dist_squared, dim=2
    )  # Shape (B, M, N), same operation due to shape

    # Compute cost matrix as average of minimum distances
    cost_matrix = (
        min_dist_pred_to_gt.mean(dim=2) + min_dist_gt_to_pred.mean(dim=2)
    ) / 2

    return cost_matrix


@MATCH_COST.register_module()
class VisBCECost(object):
    """VisibilityBCECost.
    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, vis_scores, gt_vis):
        """
        Args:
            vis_scores (Tensor): Predicted visibility scores. Shape [num_query, 10].
            gt_vis (Tensor): Ground truth visibility. Shape [num_gt, 10].
        Returns:
            torch.Tensor: vis_cost value with weight
        """
        # Expand dimensions to allow broadcasting for pairwise cost calculation
        expanded_vis_scores = vis_scores[:, None, :]  # [num_query, 1, 10]
        expanded_gt_vis = gt_vis[None, :, :]  # [1, num_gt, 10]

        # Compute BCE loss for each pair
        bce_cost = F.binary_cross_entropy_with_logits(
            expanded_vis_scores.expand(-1, gt_vis.size(0), -1),
            expanded_gt_vis.expand(vis_scores.size(0), -1, -1),
            reduction="none",
        )

        # Sum across the visibility vector dimension (axis=2)
        bce_cost = bce_cost.sum(dim=2)

        return bce_cost * self.weight
