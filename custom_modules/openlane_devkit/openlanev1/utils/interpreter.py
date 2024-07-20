# this file includes task interpreter for 3d object detection and 3d lane detection.

# for semantic output, we use grouping to get the final semantic output.

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_semantics(semantics, num_cls):
    """
    Group the 21-dim semantic output into task-specific classes.

    Parameters:
    - semantics: Tensor of shape [..., C_s], where C_s is the number of semantic classes
    - num_cls: Number of classes for the task

    Returns:
    - grouped_semantics: Tensor of shape [..., C], where C is the number of classes for the task
    """
    # consider mean, max or sum?
    grouped_semantics = semantics.view(
        *semantics.shape[:-1], num_cls, semantics.shape[-1] // num_cls
    ).max(dim=-1)[0]
    return grouped_semantics


def group_semantics_mean(semantics, num_cls):
    """
    Group the 21-dim semantic output into task-specific classes.

    Parameters:
    - semantics: Tensor of shape [..., C_s], where C_s is the number of semantic classes
    - num_cls: Number of classes for the task

    Returns:
    - grouped_semantics: Tensor of shape [..., C], where C is the number of classes for the task
    """
    # consider mean, max or sum?
    grouped_semantics = semantics.view(
        *semantics.shape[:-1], num_cls, semantics.shape[-1] // num_cls
    ).mean(dim=-1)
    return grouped_semantics


class MeanCenterNN(nn.Module):
    """A Type of Interpreter from point set to 3D bounding box.
    bbox pred format:
    (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
    where cx, cy, cz are normalized.
    """

    def __init__(self, range_min, range_max, point_set_size=10, code_size=7):
        super(MeanCenterNN, self).__init__()
        self.fc1 = nn.Linear(point_set_size * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, code_size)
        self.range_min = torch.tensor(range_min, dtype=torch.float32)
        self.range_max = torch.tensor(range_max, dtype=torch.float32)

    def normalize(self, x):
        range_min = self.range_min.to(x.device)
        range_max = self.range_max.to(x.device)
        return (x - range_min) / (range_max - range_min)

    def denormalize(self, x):
        range_min = self.range_min.to(x.device)
        range_max = self.range_max.to(x.device)
        return x * (range_max - range_min) + range_min

    def forward(self, point_set):
        """forward

        Args:
            point_set (_type_): shape (layer_out, batch_size, num_query, point_set_size, 3)

        Returns:
            _type_: regression results
        """
        # Normalize
        point_set = self.normalize(point_set)

        mean_center = torch.mean(point_set, dim=-2)
        x = torch.relu(self.fc1(point_set.view(*point_set.shape[:-2], -1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Denormalize only some part of x.
        x[..., :3] = self.denormalize(x[..., :3])

        refined_center = (x[..., :3] + mean_center) / 2
        cx, cy, cz = (
            refined_center[..., 0],
            refined_center[..., 1],
            refined_center[..., 2],
        )
        l, w, h = x[..., 3], x[..., 4], x[..., 5]
        rot_cosine, rot_sine = x[..., 6], x[..., 7]
        vx, vy = x[..., 8], x[..., 9]

        tensors = [cx, cy, w, l, cz, h, rot_cosine, rot_sine, vx, vy]
        result = torch.cat([t.unsqueeze(-1) for t in tensors], dim=-1)
        return result


# v3
class DirectionVectorBBoxInterpreterCTV(nn.Module):
    """Center to Vertex BBox Interpreter"""

    def __init__(self):
        super(DirectionVectorBBoxInterpreterCTV, self).__init__()
        # Other initializations

    def forward(self, point_set):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set, dim=-2)
        cx, cy, cz = center[..., 0], center[..., 1], center[..., 2]

        # Step 2 & 3: Calculate the direction vector and length
        direction_vector = point_set[:, :, :, 0, :2] - center[..., :2]
        zeros = torch.zeros_like(direction_vector[..., :1])
        direction_vector = torch.cat([direction_vector, zeros], dim=-1)

        unit_direction_vector = F.normalize(direction_vector, p=2, dim=-1)
        rot_sine, rot_cosine = (
            unit_direction_vector[..., 1],
            unit_direction_vector[..., 0],
        )
        perpendicular_unit_direction_vector = torch.stack(
            [
                unit_direction_vector[..., 1],
                -unit_direction_vector[..., 0],
                unit_direction_vector[..., 2],
            ],
            dim=-1,
        )

        # Project points along direction vector to find length
        proj_x = torch.sum(point_set * unit_direction_vector.unsqueeze(-2), dim=-1)
        proj_y = torch.sum(
            point_set * perpendicular_unit_direction_vector.unsqueeze(-2), dim=-1
        )
        # the length/width definition is a bit different here, it is the order of these dimension, not the name of them that defines that they are.
        length = torch.max(proj_x, dim=-1)[0] - torch.min(proj_x, dim=-1)[0]
        width = torch.max(proj_y, dim=-1)[0] - torch.min(proj_y, dim=-1)[0]
        height = (
            torch.max(point_set[..., 2], dim=-1)[0]
            - torch.min(point_set[..., 2], dim=-1)[0]
        )

        length = length.log()
        width = width.log()
        height = height.log()

        # Combine these into your bounding box representation
        bbox = torch.stack(
            [cx, cy, length, width, cz, height, rot_sine, rot_cosine], dim=-1
        )

        return bbox


# v1
class DirectionVectorBBoxInterpreter(nn.Module):
    def __init__(self):
        super(DirectionVectorBBoxInterpreter, self).__init__()
        # Other initializations

    def forward(self, point_set):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set, dim=-2)
        cx, cy, cz = center[..., 0], center[..., 1], center[..., 2]

        # Step 2 & 3: Calculate the direction vector and length
        direction_vector = point_set[:, :, :, 1, :2] - point_set[:, :, :, 0, :2]
        zeros = torch.zeros_like(direction_vector[..., :1])
        direction_vector = torch.cat([direction_vector, zeros], dim=-1)

        unit_direction_vector = F.normalize(direction_vector, p=2, dim=-1)
        rot_sine, rot_cosine = (
            unit_direction_vector[..., 1],
            unit_direction_vector[..., 0],
        )
        perpendicular_unit_direction_vector = torch.stack(
            [
                unit_direction_vector[..., 1],
                -unit_direction_vector[..., 0],
                unit_direction_vector[..., 2],
            ],
            dim=-1,
        )

        # Project points along direction vector to find length
        proj_x = torch.sum(point_set * unit_direction_vector.unsqueeze(-2), dim=-1)
        proj_y = torch.sum(
            point_set * perpendicular_unit_direction_vector.unsqueeze(-2), dim=-1
        )
        # the length/width definition is a bit different here, it is the order of these dimension, not the name of them that defines that they are.
        length = torch.max(proj_x, dim=-1)[0] - torch.min(proj_x, dim=-1)[0]
        width = torch.max(proj_y, dim=-1)[0] - torch.min(proj_y, dim=-1)[0]
        height = (
            torch.max(point_set[..., 2], dim=-1)[0]
            - torch.min(point_set[..., 2], dim=-1)[0]
        )

        length = length.log()
        width = width.log()
        height = height.log()

        # Combine these into your bounding box representation
        bbox = torch.stack(
            [cx, cy, length, width, cz, height, rot_sine, rot_cosine], dim=-1
        )

        return bbox


# v2
class DirectionVectorBBoxInterpreterEnhanced(nn.Module):
    def __init__(self):
        super(DirectionVectorBBoxInterpreterEnhanced, self).__init__()
        # Other initializations

    def forward(self, point_set):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set[..., 2:, :], dim=-2)
        cx, cy, cz = center[..., 0], center[..., 1], center[..., 2]

        # Step 2 & 3: Calculate the direction vector and length
        direction_vector = point_set[:, :, :, 1, :2] - point_set[:, :, :, 0, :2]
        zeros = torch.zeros_like(direction_vector[..., :1])
        direction_vector = torch.cat([direction_vector, zeros], dim=-1)

        unit_direction_vector = F.normalize(direction_vector, p=2, dim=-1)
        rot_sine, rot_cosine = (
            unit_direction_vector[..., 1],
            unit_direction_vector[..., 0],
        )
        perpendicular_unit_direction_vector = torch.stack(
            [
                unit_direction_vector[..., 1],
                -unit_direction_vector[..., 0],
                unit_direction_vector[..., 2],
            ],
            dim=-1,
        )

        # Project points along direction vector to find length
        proj_x = torch.sum(
            point_set[..., 2:, :] * unit_direction_vector.unsqueeze(-2), dim=-1
        )
        proj_y = torch.sum(
            point_set[..., 2:, :] * perpendicular_unit_direction_vector.unsqueeze(-2),
            dim=-1,
        )
        # the length/width definition is a bit different here, it is the order of these dimension, not the name of them that defines that they are.
        length = torch.max(proj_x, dim=-1)[0] - torch.min(proj_x, dim=-1)[0]
        width = torch.max(proj_y, dim=-1)[0] - torch.min(proj_y, dim=-1)[0]
        height = (
            torch.max(point_set[..., 2:, 2], dim=-1)[0]
            - torch.min(point_set[..., 2:, 2], dim=-1)[0]
        )

        length = length.log()
        width = width.log()
        height = height.log()

        # Combine these into your bounding box representation
        bbox = torch.stack(
            [cx, cy, length, width, cz, height, rot_sine, rot_cosine], dim=-1
        )

        return bbox


# v4 soft min max
class DirectionVectorBBoxInterpreterTopkCTV(nn.Module):
    """Center to Vertex BBox Interpreter"""

    def __init__(self, k=3, weight_scheme="exponential"):
        self.k = k
        self.weight_scheme = weight_scheme
        # Define the weights
        if weight_scheme == "linear":
            self.weights = torch.linspace(k, 1, k)
        elif weight_scheme == "exponential":
            self.weights = torch.pow(2, torch.arange(k, 0, -1))
        else:
            raise ValueError(
                "Unsupported weight_scheme. Use 'linear' or 'exponential'."
            )
        self.weights = self.weights / torch.sum(self.weights)
        self.weights.requires_grad = False

        super(DirectionVectorBBoxInterpreterTopkCTV, self).__init__()
        # Other initializations

    def forward(self, point_set):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set, dim=-2)
        cx, cy, cz = center[..., 0], center[..., 1], center[..., 2]

        # Step 2 & 3: Calculate the direction vector and length
        direction_vector = point_set[:, :, :, 0, :2] - center[..., :2]
        zeros = torch.zeros_like(direction_vector[..., :1])
        direction_vector = torch.cat([direction_vector, zeros], dim=-1)

        unit_direction_vector = F.normalize(direction_vector, p=2, dim=-1)
        rot_sine, rot_cosine = (
            unit_direction_vector[..., 1],
            unit_direction_vector[..., 0],
        )
        perpendicular_unit_direction_vector = torch.stack(
            [
                unit_direction_vector[..., 1],
                -unit_direction_vector[..., 0],
                unit_direction_vector[..., 2],
            ],
            dim=-1,
        )

        # Project points along direction vector to find length
        proj_x = torch.sum(point_set * unit_direction_vector.unsqueeze(-2), dim=-1)
        proj_y = torch.sum(
            point_set * perpendicular_unit_direction_vector.unsqueeze(-2), dim=-1
        )
        # the length/width definition is a bit different here, it is the order of these dimension, not the name of them that defines that they are.

        # topk conversion
        to_convert = [proj_x, proj_y, point_set[..., 2]]

        converted = []
        self.weights = self.weights.to(point_set.device)
        for t in to_convert:
            topk_max, _ = torch.topk(t, self.k, dim=-1)
            topk_min, _ = torch.topk(-t, self.k, dim=-1)
            topk_dims = topk_max + topk_min
            weighted_topk_dim = torch.sum(topk_dims * self.weights, dim=-1)
            converted.append(weighted_topk_dim)

        length, width, height = converted

        length = length.log()
        width = width.log()
        height = height.log()

        # Combine these into your bounding box representation
        bbox = torch.stack(
            [cx, cy, length, width, cz, height, rot_sine, rot_cosine], dim=-1
        )

        return bbox


# v5 std


class CTVHeading(nn.Module):
    """Center to Vertex BBox Interpreter"""

    def __init__(self):
        super(CTVHeading, self).__init__()
        # Other initializations

    def forward(self, point_set):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set, dim=-2)

        # Step 2 & 3: Calculate the direction vector and length
        direction_vector = point_set[:, :, :, 0, :2] - center[..., :2]
        zeros = torch.zeros_like(direction_vector[..., :1])
        direction_vector = torch.cat([direction_vector, zeros], dim=-1)

        unit_direction_vector = F.normalize(direction_vector, p=2, dim=-1)
        rot_sine, rot_cosine = (
            unit_direction_vector[..., 1],
            unit_direction_vector[..., 0],
        )
        # Combine these into your bounding box representation
        heading = torch.stack([rot_sine, rot_cosine], dim=-1)

        return heading


class VectorField2DHeading(nn.Module):
    """Center to Vertex BBox Interpreter"""

    def __init__(self):
        super(VectorField2DHeading, self).__init__()
        # Other initializations

    def forward(self, vector_field):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 5]
        # vector field is essentially a point set.
        heading = torch.mean(vector_field[..., 3:5], dim=-2)
        return heading


# v5 part2
class DirectionVectorStdBBoxInterpreter(nn.Module):
    """Center to Vertex BBox Interpreter"""

    def __init__(self):
        super(DirectionVectorStdBBoxInterpreter, self).__init__()
        self.moment_transfer = nn.Parameter(torch.zeros(3))
        self.moment_mul = 0.1
        # Other initializations

    def forward(self, point_set, heading):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]
        rot_sine, rot_cosine = heading[..., 0], heading[..., 1]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set, dim=-2)
        cx, cy, cz = center[..., 0], center[..., 1], center[..., 2]

        unit_direction_vector = torch.stack([rot_cosine, rot_sine], dim=-1)
        zeros = torch.zeros_like(unit_direction_vector[..., :1])
        unit_direction_vector = torch.cat([unit_direction_vector, zeros], dim=-1)

        perpendicular_unit_direction_vector = torch.stack(
            [
                unit_direction_vector[..., 1],
                -unit_direction_vector[..., 0],
                unit_direction_vector[..., 2],
            ],
            dim=-1,
        )

        # Project points along direction vector to find length
        proj_x = torch.sum(point_set * unit_direction_vector.unsqueeze(-2), dim=-1)
        proj_y = torch.sum(
            point_set * perpendicular_unit_direction_vector.unsqueeze(-2), dim=-1
        )
        proj_x_mean = torch.mean(proj_x, dim=-1)
        proj_y_mean = torch.mean(proj_y, dim=-1)
        z_mean = torch.mean(point_set[..., 2], dim=-1)
        proj_x_std = torch.std(proj_x - proj_x_mean[..., None], dim=-1)
        proj_y_std = torch.std(proj_y - proj_y_mean[..., None], dim=-1)
        z_std = torch.std(point_set[..., 2] - z_mean[..., None], dim=-1)
        moment_transfer = (self.moment_transfer * self.moment_mul) + (
            self.moment_transfer.detach() * (1 - self.moment_mul)
        )

        l_t, w_t, h_t = moment_transfer[0], moment_transfer[1], moment_transfer[2]

        length = proj_x_std * torch.exp(l_t)
        width = proj_y_std * torch.exp(w_t)
        height = z_std * torch.exp(h_t)

        length = length.log()
        width = width.log()
        height = height.log()

        # Combine these into your bounding box representation
        bbox = torch.stack(
            [cx, cy, length, width, cz, height, rot_sine, rot_cosine], dim=-1
        )

        return bbox


class DirectionVectorStdBBoxInterpreterHeadingFix(nn.Module):
    """Center to Vertex BBox Interpreter"""

    def __init__(self):
        super(DirectionVectorStdBBoxInterpreterHeadingFix, self).__init__()
        self.moment_transfer = nn.Parameter(torch.zeros(3))
        self.moment_mul = 0.1
        # Other initializations

    def forward(self, point_set, heading):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]
        rot_sine, rot_cosine = heading[..., 0], heading[..., 1]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set, dim=-2)
        cx, cy, cz = center[..., 0], center[..., 1], center[..., 2]

        unit_direction_vector = torch.stack([rot_cosine, rot_sine], dim=-1)
        unit_direction_vector = F.normalize(unit_direction_vector, p=2, dim=-1)
        zeros = torch.zeros_like(unit_direction_vector[..., :1])
        unit_direction_vector = torch.cat([unit_direction_vector, zeros], dim=-1)

        perpendicular_unit_direction_vector = torch.stack(
            [
                unit_direction_vector[..., 1],
                -unit_direction_vector[..., 0],
                unit_direction_vector[..., 2],
            ],
            dim=-1,
        )

        # Project points along direction vector to find length
        proj_x = torch.sum(point_set * unit_direction_vector.unsqueeze(-2), dim=-1)
        proj_y = torch.sum(
            point_set * perpendicular_unit_direction_vector.unsqueeze(-2), dim=-1
        )
        proj_x_mean = torch.mean(proj_x, dim=-1)
        proj_y_mean = torch.mean(proj_y, dim=-1)
        z_mean = torch.mean(point_set[..., 2], dim=-1)
        proj_x_std = torch.std(proj_x - proj_x_mean[..., None], dim=-1)
        proj_y_std = torch.std(proj_y - proj_y_mean[..., None], dim=-1)
        z_std = torch.std(point_set[..., 2] - z_mean[..., None], dim=-1)
        moment_transfer = (self.moment_transfer * self.moment_mul) + (
            self.moment_transfer.detach() * (1 - self.moment_mul)
        )

        l_t, w_t, h_t = moment_transfer[0], moment_transfer[1], moment_transfer[2]

        length = proj_x_std * torch.exp(l_t)
        width = proj_y_std * torch.exp(w_t)
        height = z_std * torch.exp(h_t)

        length = length.log()
        width = width.log()
        height = height.log()

        # Combine these into your bounding box representation
        bbox = torch.stack(
            [cx, cy, length, width, cz, height, heading[..., 0], heading[..., 1]], dim=-1
        )

        return bbox

class DirectionVectorMinMaxBBoxInterpreter(nn.Module):

    def __init__(self):
        super(DirectionVectorMinMaxBBoxInterpreter, self).__init__()
        # Other initializations

    def forward(self, point_set, heading):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]
        rot_sine, rot_cosine = heading[..., 0], heading[..., 1]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set, dim=-2)
        cx, cy, cz = center[..., 0], center[..., 1], center[..., 2]

        unit_direction_vector = torch.stack([rot_cosine, rot_sine], dim=-1)
        zeros = torch.zeros_like(unit_direction_vector[..., :1])
        unit_direction_vector = torch.cat([unit_direction_vector, zeros], dim=-1)

        perpendicular_unit_direction_vector = torch.stack(
            [
                unit_direction_vector[..., 1],
                -unit_direction_vector[..., 0],
                unit_direction_vector[..., 2],
            ],
            dim=-1,
        )

        # Project points along direction vector to find length
        proj_x = torch.sum(point_set * unit_direction_vector.unsqueeze(-2), dim=-1)
        proj_y = torch.sum(
            point_set * perpendicular_unit_direction_vector.unsqueeze(-2), dim=-1
        )
        length = torch.max(proj_x, dim=-1)[0] - torch.min(proj_x, dim=-1)[0]
        width = torch.max(proj_y, dim=-1)[0] - torch.min(proj_y, dim=-1)[0]
        height = (
            torch.max(point_set[..., 2:, 2], dim=-1)[0]
            - torch.min(point_set[..., 2:, 2], dim=-1)[0]
        )

        length = length.log()
        width = width.log()
        height = height.log()

        # Combine these into your bounding box representation
        bbox = torch.stack(
            [cx, cy, length, width, cz, height, rot_sine, rot_cosine], dim=-1
        )

        return bbox


# v8 p2
class DirectionVectorBBoxInterpreterNoHeading(nn.Module):
    """Center to Vertex BBox Interpreter"""

    def __init__(self):
        super(DirectionVectorBBoxInterpreterNoHeading, self).__init__()
        # self.moment_transfer = nn.Parameter(torch.zeros(3))
        # self.moment_mul = 0.1
        # Other initializations

    def forward(self, point_set, heading):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]
        rot_sine, rot_cosine = heading[..., 0], heading[..., 1]

        # Step 1: Calculate the center of the bounding box
        center = torch.mean(point_set, dim=-2)
        cx, cy, cz = center[..., 0], center[..., 1], center[..., 2]

        unit_direction_vector = torch.stack([rot_cosine, rot_sine], dim=-1)
        zeros = torch.zeros_like(unit_direction_vector[..., :1])
        unit_direction_vector = torch.cat([unit_direction_vector, zeros], dim=-1)

        perpendicular_unit_direction_vector = torch.stack(
            [
                unit_direction_vector[..., 1],
                -unit_direction_vector[..., 0],
                unit_direction_vector[..., 2],
            ],
            dim=-1,
        )

        # Project points along direction vector to find length
        proj_x = torch.sum(
            point_set[..., 2:, :] * unit_direction_vector.unsqueeze(-2), dim=-1
        )
        proj_y = torch.sum(
            point_set[..., 2:, :] * perpendicular_unit_direction_vector.unsqueeze(-2),
            dim=-1,
        )
        # the length/width definition is a bit different here, it is the order of these dimension, not the name of them that defines that they are.
        length = torch.max(proj_x, dim=-1)[0] - torch.min(proj_x, dim=-1)[0]
        width = torch.max(proj_y, dim=-1)[0] - torch.min(proj_y, dim=-1)[0]
        height = (
            torch.max(point_set[..., 2:, 2], dim=-1)[0]
            - torch.min(point_set[..., 2:, 2], dim=-1)[0]
        )

        # Project points along direction vector to find length
        # proj_x = torch.sum(point_set * unit_direction_vector.unsqueeze(-2), dim=-1)
        # proj_y = torch.sum(
        # point_set * perpendicular_unit_direction_vector.unsqueeze(-2), dim=-1
        # )
        # proj_x_mean = torch.mean(proj_x, dim=-1)
        # proj_y_mean = torch.mean(proj_y, dim=-1)
        # z_mean = torch.mean(point_set[..., 2], dim=-1)
        # proj_x_std = torch.std(proj_x - proj_x_mean[..., None], dim=-1)
        # proj_y_std = torch.std(proj_y - proj_y_mean[..., None], dim=-1)
        # z_std = torch.std(point_set[..., 2] - z_mean[..., None], dim=-1)
        # moment_transfer = (self.moment_transfer * self.moment_mul) + (
        # self.moment_transfer.detach() * (1 - self.moment_mul)
        # )
        #
        # l_t, w_t, h_t = moment_transfer[0], moment_transfer[1], moment_transfer[2]
        #
        # length = proj_x_std * torch.exp(l_t)
        # width = proj_y_std * torch.exp(w_t)
        # height = z_std * torch.exp(h_t)

        length = length.log()
        width = width.log()
        height = height.log()

        # Combine these into your bounding box representation
        bbox = torch.stack(
            [cx, cy, length, width, cz, height, rot_sine, rot_cosine], dim=-1
        )

        return bbox


class PointSortInterpreter(nn.Module):
    """Sort by x within each set in an increasing manner for each point set query"""

    def __init__(self):
        super(PointSortInterpreter, self).__init__()

    def forward(self, point_set, field_dims=3):
        # Assume point_set is of shape [layer, batch_size, num_queries, num_points_per_set, 3]

        # Extract the x-coordinates
        x_coords = point_set[..., 0]

        # Sort the x-coordinates along the num_points_per_set dimension
        sorted_x, sorted_indices = torch.sort(x_coords, dim=3)

        # Expand the indices to use for all coordinates (x, y, z, ...)
        expanded_indices = sorted_indices.unsqueeze(-1).expand(
            -1, -1, -1, -1, field_dims
        )

        # Sort all points according to the sorted indices
        sorted_point_set = torch.gather(point_set, 3, expanded_indices)

        return sorted_point_set
