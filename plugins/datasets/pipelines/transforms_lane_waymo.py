import numpy as np
from scipy.interpolate import interp1d

import mmcv
from mmdet.datasets import PIPELINES

from openlanev1.utils import anchor


@PIPELINES.register_module()
class ConvertLaneToAnchors_FGAssumpWaymo:
    """
    Convert Lane Annotations to anchor representation, with flat ground assumption.

    before doing that, this functions involves ground space to ground plane, and this is where ground plane assumption happend.

    TODO: We should break this assumption one day or another!

    Add: gt_anchor_lanes{
        assign_ids, gt_anchors, visibility_vectors, category_ids
    }
    Update: None
    """

    def __init__(
        self,
        anchor_params_dict=None,
        prune_x_range=(0, 200),
        x_ref=5,
        y_off_std=np.ones(10),
        z_std=np.ones(10),
        is_calc_mode=False,
    ):
        # these parameters are used to generate anchors
        self.y_min, self.y_max, self.y_ratio = anchor_params_dict["y_range"]
        self.x_min, self.x_max = anchor_params_dict["x_range"]
        self.prune_x_min, self.prune_x_max = prune_x_range
        self.x_ref = x_ref
        self.anchor_x_steps = anchor_params_dict["anchor_x_steps"]

        self.anchor_grid_x = anchor.AnchorGridWaymo(**anchor_params_dict).anchor_grid_y

        # for gt anchor generation standardize
        self._y_off_std, self._z_std = np.array(y_off_std), np.array(z_std)

        # Initialize running statistics for dynamic calculations
        self.sample_count = np.zeros(len(self.anchor_x_steps))
        self.running_mean_y = np.zeros(len(self.anchor_x_steps))
        self.running_mean_z = np.zeros(len(self.anchor_x_steps))
        self.running_ssq_y = np.zeros(len(self.anchor_x_steps))
        self.running_ssq_z = np.zeros(len(self.anchor_x_steps))
        # calculate std for x_off and z dynamically
        self.is_calc_mode = is_calc_mode

    def __call__(self, results):
        """
        this involved three steps:
        1. do a deep copy of the lane annotations
        2. prepare ground space to ground plane transformation matrix
        3. convert 3d lanes onto the 'ground plane'
        4. generate anchors
        """
        # get non-anchor lane annotations
        gt_lane_pts = results["gt_lanes"]["lane_pts"]
        gt_lane_visibility = results["gt_lanes"]["lane_visibility"]
        gt_lane_category = results["gt_lanes"]["lane_category"]
        # do deep copy
        gt_lane_pts, gt_lane_visibility, gt_lane_category = (
            [np.copy(gt_lane_pts[k]) for k in range(len(gt_lane_pts))],
            [np.copy(gt_lane_visibility[k]) for k in range(len(gt_lane_visibility))],
            [np.copy(gt_lane_category[k]) for k in range(len(gt_lane_category))],
        )
        # name alias
        gt_lanes, gt_visibility, gt_category = (
            gt_lane_pts,
            gt_lane_visibility,
            gt_lane_category,
        )

        # camera parameters for anchor generation
        cam_K = results["calib"]["intrinsics"][0]
        cam_E = results["calib"]["extrinsics"][0]
        # g2im stands for ground to image
        P_g2im = self.projection_g2im_extrinsic(
            cam_E, cam_K
        )  # ground space to camera plane
        H_g2im = self.homograpthy_g2im_extrinsic(
            cam_E, cam_K
        )  # ground plane to camera plane
        H_im2g = np.linalg.inv(H_g2im)  # camera plane to ground plane
        P_g2gflat = np.matmul(H_im2g, P_g2im)  # ground space to ground plane

        # convert 3d lanes to ground plane
        for lane in gt_lanes:
            # from [x,y,z,1](ground space) to [u,v,1](ground plane)
            lane_gflat_x, lane_gflat_y = self.projective_transformation(
                P_g2gflat, lane[:, 0], lane[:, 1], lane[:, 2]
            )
            lane[:, 0] = lane_gflat_x
            lane[:, 1] = lane_gflat_y

        # generate anchors
        gt_anchors = []
        ass_ids = []
        visibility_vectors = []
        category_ids = []
        for i in range(len(gt_lanes)):
            # convert gt label to anchor label
            # consider individual out-of-range interpolation still visible
            (
                ass_id,
                y_off_values,
                z_values,
                visibility_vec,
            ) = self.convert_label_to_anchor(gt_lanes[i])
            if ass_id >= 0:
                gt_anchors.append(np.vstack([y_off_values, z_values]).T)
                ass_ids.append(ass_id)
                visibility_vectors.append(visibility_vec)
                category_ids.append(gt_category[i])
        # normalize x anad z, in replacement of normalize_lane_label
        if not self.is_calc_mode:
            for lane in gt_anchors:
                lane[:, 0] = np.divide(lane[:, 0], self._y_off_std)
                lane[:, 1] = np.divide(lane[:, 1], self._z_std)
        else:
            if gt_anchors:
                # dynamicly calculate x_off_std and z_std
                lane_y_off = np.vstack([lane[:, 0] for lane in gt_anchors])
                lane_z = np.vstack([lane[:, 1] for lane in gt_anchors])
                visibility_weights = np.vstack(visibility_vectors)

                # Update running statistics for lane_x_off and lane_z
                self.update_statistics(lane_y_off, lane_z, visibility_weights)

                # Normalize x and z
                # for lane in gt_anchors:
                #     for i, (x_std, z_std) in enumerate(zip(self._x_off_std, self._z_std)):
                #         lane[:, 0][i] = lane[:, 0][i] / x_std
                #         lane[:, 1][i] = lane[:, 1][i] / z_std

        results.update(
            {
                "gt_anchor_lanes": {
                    "ass_ids": ass_ids,
                    "gt_anchors": gt_anchors,
                    "visibility_vectors": visibility_vectors,
                    "category_ids": category_ids,
                }
            }
        )
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            # parameters
            f"is_calc_mode={self.is_calc_mode}, "
            f"lane_x_off_std={self._y_off_std}, "
            f"lane_z_std={self._z_std}"
            f")"
        )
        return repr_str

    # Update running statistics for lane_x_off and lane_z
    def update_statistics_old(self, lane_x_off, lane_z, visibility_weights):
        # shape (N, 10), N is the count of lanes, 10 is the anchor number
        # expected output: (10) for _x_off_std & _z_std.
        for x, z, weight in zip(lane_x_off, lane_z, visibility_weights):
            for i, w in enumerate(weight):
                # if w > 0:  # Update statistics only for valid anchor points
                self.sample_count[i] += w
                delta_x = x[i] - self.running_mean_y[i]
                delta_z = z[i] - self.running_mean_z[i]

                self.running_mean_y[i] += w * delta_x / self.sample_count[i]
                self.running_mean_z[i] += w * delta_z / self.sample_count[i]

                self.running_ssq_y[i] += w * (x[i] ** 2)
                self.running_ssq_z[i] += w * (z[i] ** 2)

        # Calculate standard deviations from running weighted sum of squares
        self._y_off_std = np.sqrt(
            self.running_ssq_y / self.sample_count - self.running_mean_y**2
        )
        self._z_std = np.sqrt(
            self.running_ssq_z / self.sample_count - self.running_mean_z**2
        )

    # weighted RMS
    def update_statistics(self, lane_y_off, lane_z, visibility_weights):
        # shape (N, 10), N is the count of lanes, 10 is the anchor number
        # expected output: (10) for _x_off_std & _z_std.
        for y, z, weight in zip(lane_y_off, lane_z, visibility_weights):
            for i, w in enumerate(weight):
                # if w > 0:  # Update statistics only for valid anchor points
                self.sample_count[i] += w
                # delta_x = x[i] - self.running_mean_x[i]
                # delta_z = z[i] - self.running_mean_z[i]

                # self.running_mean_x[i] += w * delta_x / self.sample_count[i]
                # self.running_mean_z[i] += w * delta_z / self.sample_count[i]

                self.running_ssq_y[i] += w * (y[i] ** 2)
                self.running_ssq_z[i] += w * (z[i] ** 2)

        # Calculate standard deviations from running weighted sum of squares
        self._y_off_std = np.sqrt(self.running_ssq_y / self.sample_count)
        self._z_std = np.sqrt(self.running_ssq_z / self.sample_count)

    # flat ground assumption
    @staticmethod
    def homograpthy_g2im_extrinsic(E, K):
        """E: extrinsic matrix, 4*4
        H: homography matrix, 3*3"""
        # E_inv stands for extrinsic inverse,
        # this is a homogeneous matrix, 4*4
        # extrinsic is defied as camera to ground, so ground to camera is inverse, q: camera to ground is actually the coordinate of camera in ground space? a: yes
        E_inv = np.linalg.inv(E)[0:3, :]
        # g2c stands for ground to camera
        # this is where the magic happens, we only need the first 3 rows, and the first 3 columns, and the magic is <flat ground assumption>.
        H_g2c = E_inv[:, [0, 1, 3]]
        # to image(gournd) = to image(camera) * to camera(ground)
        H_g2im = np.matmul(K, H_g2c)
        return H_g2im

    @staticmethod
    def projection_g2im_extrinsic(E, K):
        """P: projection matrix, 3*4"""
        E_inv = np.linalg.inv(E)[0:3, :]  # ground2cam
        P_g2im = np.matmul(K, E_inv)
        return P_g2im

    @staticmethod
    def projective_transformation(Matrix, x, y, z):
        """
        Helper function to transform coordinates defined by transformation matrix

        Args:
                Matrix (multi dim - array): 3x4 projection matrix
                x (array): original x coordinates
                y (array): original y coordinates
                z (array): original z coordinates
        """
        ones = np.ones((1, len(z)))
        coordinates = np.vstack((x, y, z, ones))
        trans = np.matmul(Matrix, coordinates)

        x_vals = trans[0, :] / trans[2, :]
        y_vals = trans[1, :] / trans[2, :]
        return x_vals, y_vals

    def convert_label_to_anchor(self, laneline_gt):
        """
            Convert a set of ground-truth lane points to the format of network anchor representation.

            All the given laneline only include visible points. The interpolated points will be marked invisible
        :param laneline_gt: a list of arrays where each array is a set of point coordinates in [x, y, z]
        :return: ass_id: the column id of current lane in anchor representation
                    x_off_values: current lane's x offset from it associated anchor column
                    z_values: current lane's z value in ground coordinates
        """

        # For ground-truth in ground coordinates (Apollo Sim)
        gt_lane_3d = laneline_gt

        # prune out points not in valid range, requires additional points to interpolate better
        # prune out-of-range points after transforming to flat ground space, update visibility vector
        valid_indices = np.logical_and(
            np.logical_and(
                gt_lane_3d[:, 0] > self.prune_x_min, gt_lane_3d[:, 0] < self.prune_x_max
            ),
            np.logical_and(
                gt_lane_3d[:, 1] > self.y_ratio * self.y_min,
                gt_lane_3d[:, 1] < self.y_ratio * self.y_max,
            ),
        )
        gt_lane_3d = gt_lane_3d[valid_indices, ...]
        # use more restricted range to determine deletion or not
        if (
            gt_lane_3d.shape[0] < 2
            or np.sum(
                np.logical_and(
                    gt_lane_3d[:, 1] > self.y_min, gt_lane_3d[:, 1] < self.y_max
                )
            )
            < 2
        ):
            return -1, np.array([]), np.array([]), np.array([])

        # only keep the portion y is monotonically increasing above a threshold, to prune those super close points
        gt_lane_3d = self.make_lane_x_mono_inc(gt_lane_3d)
        if gt_lane_3d.shape[0] < 2:
            return -1, np.array([]), np.array([]), np.array([])

        # ignore GT ends before y_ref, for those start at y > y_ref, use its interpolated value at y_ref for association
        if gt_lane_3d[-1, 0] < self.x_ref:
            return -1, np.array([]), np.array([]), np.array([])

        # resample ground-truth laneline at anchor y steps
        y_values, z_values, visibility_vec = self.resample_laneline_in_x(
            gt_lane_3d, self.anchor_x_steps, out_vis=True
        )

        if np.sum(visibility_vec) < 2:
            return -1, np.array([]), np.array([]), np.array([])

        # decide association at visible offset locations
        ass_id = np.argmin(
            np.linalg.norm(
                np.multiply(self.anchor_grid_x - y_values, visibility_vec), axis=1
            )
        )
        # compute offset values
        y_off_values = y_values - self.anchor_grid_x[ass_id]

        return ass_id, y_off_values, z_values, visibility_vec

    @staticmethod
    def make_lane_x_mono_inc(lane):
        """
            Due to lose of height dim, projected lanes to flat ground plane may not have monotonically increasing y.
            This function trace the y with monotonically increasing y, and output a pruned lane
        :param lane:
        :return:
        """
        idx2del = []
        max_x = lane[0, 0]
        for i in range(1, lane.shape[0]):
            # hard-coded a smallest step, so the far-away near horizontal tail can be pruned
            if lane[i, 0] <= max_x + 3:
                idx2del.append(i)
            else:
                max_x = lane[i, 0]
        lane = np.delete(lane, idx2del, axis=0)
        return lane

    @staticmethod
    def resample_laneline_in_x(input_lane, y_steps, out_vis=False):
        """
            Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
        :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                        It requires y values of input lane in ascending order
        :param y_steps: a vector of steps in y
        :param out_vis: whether to output visibility indicator which only depends on input y range
        :return:
        """

        # at least two points are included
        assert input_lane.shape[0] >= 2

        x_min = np.min(input_lane[:, 0]) - 5
        x_max = np.max(input_lane[:, 0]) + 5

        if input_lane.shape[1] < 3:
            input_lane = np.concatenate(
                [input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)],
                axis=1,
            )

        f_y = interp1d(input_lane[:, 1], input_lane[:, 1], fill_value="extrapolate")
        f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

        y_values = f_y(y_steps)
        z_values = f_z(y_steps)

        if out_vis:
            output_visibility = np.logical_and(y_steps >= x_min, y_steps <= x_max)
            return y_values, z_values, output_visibility.astype(np.float32) + 1e-9
        return y_values, z_values


@PIPELINES.register_module()
class GenerateAnchorGroundTruthWaymo:
    def __init__(self, anchor_params_dict, num_category_toggle):
        self.anchor_grid = anchor.AnchorGridWaymo(**anchor_params_dict)
        self.anchor_num = self.anchor_grid.anchor_num
        self.num_types = 1
        self.num_x_steps = self.anchor_grid.num_x_steps
        self.num_category = 21 if num_category_toggle else 22  # merge road edge
        self.anchor_dim = 3 * self.num_x_steps + self.num_category

    def __call__(self, results):
        gt_anchor = np.zeros(
            [self.anchor_num, self.num_types, self.anchor_dim], dtype=np.float32
        )
        gt_anchor[:, :, self.anchor_dim - self.num_category] = 1.0
        gt_lanes = results["gt_anchor_lanes"]["gt_anchors"]
        gt_vis_inds = results["gt_anchor_lanes"]["visibility_vectors"]
        gt_category_3d = results["gt_anchor_lanes"]["category_ids"]

        for i in range(len(gt_lanes)):
            ass_id = results["gt_anchor_lanes"]["ass_ids"][i]
            y_off_values = gt_lanes[i][:, 0]
            z_values = gt_lanes[i][:, 1]
            visibility = gt_vis_inds[i]
            # assign anchor tensor values
            gt_anchor[ass_id, 0, 0 : self.num_x_steps] = y_off_values
            gt_anchor[ass_id, 0, self.num_x_steps : 2 * self.num_x_steps] = z_values
            gt_anchor[
                ass_id, 0, 2 * self.num_x_steps : 3 * self.num_x_steps
            ] = visibility

            # gt_anchor[ass_id, 0, -1] = 1.0
            gt_anchor[ass_id, 0, self.anchor_dim - self.num_category] = 0.0
            gt_anchor[
                ass_id, 0, self.anchor_dim - self.num_category + gt_category_3d[i]
            ] = 1.0

        results["gt_anchor_lanes"].update({"packed": gt_anchor})
        return results

    def __repr__(self) -> str:
        pass
