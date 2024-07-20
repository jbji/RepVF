import numpy as np


class AnchorGrid:
    """
    Anchor_x_steps = ipm_w//8 = 16 by default, ipm mechanism is removed.
    """

    def __init__(
        self,
        x_range=(-10, 10, 3),
        y_range=(3, 103),
        anchor_num_before_shear=16,
        anchor_y_steps=[5, 10, 15, 20, 30, 40, 50, 60, 80, 100],
    ) -> None:
        self.anchor_num_before_shear = anchor_num_before_shear
        self.x_min, self.x_max, self.x_ratio = x_range
        self.y_min, self.y_max = y_range
        self.anchor_y_steps = anchor_y_steps

        self.initialize_anchor()
        pass

    def initialize_anchor(self):
        """require:
        self.anchor_y_steps, self.anchor_x
        self.ipm_w
        self.x_min, self.x_max, self.anchor_num_before_shear
        """
        self.anchor_y_steps = np.array(self.anchor_y_steps)
        self.anchor_x_steps = np.linspace(
            self.x_min, self.x_max, self.anchor_num_before_shear, endpoint=True
        )
        self.num_y_steps = len(self.anchor_y_steps)

        # compute anchor grid with different far center points
        # currently, anchor grid consists of [center, left-sheared, right-sheared] concatenated
        self.anchor_num = self.anchor_num_before_shear * 7
        self.anchor_grid_x = np.repeat(
            np.expand_dims(self.anchor_x_steps, axis=1), self.num_y_steps, axis=1
        )  # center
        anchor_grid_y = np.repeat(
            np.expand_dims(self.anchor_y_steps, axis=0),
            self.anchor_num_before_shear,
            axis=0,
        )

        # Part 1: 10 degree shear
        x2y_ratio = self.x_min / (
            self.y_max - self.y_min
        )  # x change per unit y change (for left-sheared anchors)
        anchor_grid_x_left_10 = (
            anchor_grid_y - self.y_min
        ) * x2y_ratio + self.anchor_grid_x
        # right-sheared anchors are symmetrical to left-sheared ones
        anchor_grid_x_right_10 = np.flip(-anchor_grid_x_left_10, axis=0)

        # Part 2: 20 degree shear
        x2y_ratio = (self.x_min - self.x_max) / (
            self.y_max - self.y_min
        )  # x change per unit y change (for left-sheared anchors)
        anchor_grid_x_left_20 = (
            anchor_grid_y - self.y_min
        ) * x2y_ratio + self.anchor_grid_x
        # right-sheared anchors are symmetrical to left-sheared ones
        anchor_grid_x_right_20 = np.flip(-anchor_grid_x_left_20, axis=0)

        # Part 3: 40 degree shear
        x2y_ratio = (
            2.0 * (self.x_min - self.x_max) / (self.y_max - self.y_min)
        )  # x change per unit y change (for left-sheared anchors)
        anchor_grid_x_left_40 = (
            anchor_grid_y - self.y_min
        ) * x2y_ratio + self.anchor_grid_x
        # right-sheared anchors are symmetrical to left-sheared ones
        anchor_grid_x_right_40 = np.flip(-anchor_grid_x_left_40, axis=0)

        # concat the three parts
        self.anchor_grid_x = np.concatenate(
            (
                self.anchor_grid_x,
                anchor_grid_x_left_10,
                anchor_grid_x_right_10,
                anchor_grid_x_left_20,
                anchor_grid_x_right_20,
                anchor_grid_x_left_40,
                anchor_grid_x_right_40,
            ),
            axis=0,
        )

    @property
    def anchor_grid_2d(self):
        """
        shape: [anchor_num, num_y_steps, 2]
        """
        # anchor_grid_x_shape [anchor_num, num_y_steps] = [112,10]
        # anchor_y_steps = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]

        anchor_grid_2d = np.concatenate(
            [
                self.anchor_grid_x[:, :, np.newaxis],
                self.anchor_grid_y[:, :, np.newaxis],
            ],
            axis=-1,
        )
        return anchor_grid_2d

    @property
    def anchor_grid_y(self):
        return np.repeat(
            np.expand_dims(self.anchor_y_steps, axis=0), self.anchor_num, axis=0
        )  # [anchor_num, num_y_steps]

    @property
    def anchor_grid_3d_flat(self):
        """
        shape: [anchor_num, num_y_steps, 3]
        """
        anchor_grid_2d = self.anchor_grid_2d
        anchor_grid_3d = np.zeros((self.anchor_num, self.num_y_steps, 3))
        anchor_grid_3d[:, :, :2] = anchor_grid_2d
        return anchor_grid_3d


class AnchorGridWaymo:
    def __init__(
        self,
        x_range=(3, 103),
        y_range=(-10, 10, 3),
        anchor_num_before_shear=16,
        anchor_x_steps=[5, 10, 15, 20, 30, 40, 50, 60, 80, 100],
    ) -> None:
        self.anchor_num_before_shear = anchor_num_before_shear
        self.y_min, self.y_max, self.y_ratio = y_range
        self.x_min, self.x_max = x_range
        self.anchor_x_steps = anchor_x_steps

        self.initialize_anchor()

    def initialize_anchor(self):
        self.anchor_x_steps = np.array(self.anchor_x_steps)
        self.anchor_y_steps = np.linspace(
            self.y_min, self.y_max, self.anchor_num_before_shear, endpoint=True
        )
        self.num_x_steps = len(self.anchor_x_steps)

        self.anchor_num = self.anchor_num_before_shear * 7
        self.anchor_grid_y = np.repeat(
            np.expand_dims(self.anchor_y_steps, axis=1), self.num_x_steps, axis=1
        )
        anchor_grid_x = np.repeat(
            np.expand_dims(self.anchor_x_steps, axis=0),
            self.anchor_num_before_shear,
            axis=0,
        )

        y2x_ratio = self.y_min / (self.x_max - self.x_min)
        anchor_grid_y_up_10 = (
            anchor_grid_x - self.x_min
        ) * y2x_ratio + self.anchor_grid_y
        anchor_grid_y_down_10 = np.flip(-anchor_grid_y_up_10, axis=0)

        y2x_ratio = (self.y_min - self.y_max) / (self.x_max - self.x_min)
        anchor_grid_y_up_20 = (
            anchor_grid_x - self.x_min
        ) * y2x_ratio + self.anchor_grid_y
        anchor_grid_y_down_20 = np.flip(-anchor_grid_y_up_20, axis=0)

        y2x_ratio = 2.0 * (self.y_min - self.y_max) / (self.x_max - self.x_min)
        anchor_grid_y_up_40 = (
            anchor_grid_x - self.x_min
        ) * y2x_ratio + self.anchor_grid_y
        anchor_grid_y_down_40 = np.flip(-anchor_grid_y_up_40, axis=0)

        self.anchor_grid_y = np.concatenate(
            (
                self.anchor_grid_y,
                anchor_grid_y_up_10,
                anchor_grid_y_down_10,
                anchor_grid_y_up_20,
                anchor_grid_y_down_20,
                anchor_grid_y_up_40,
                anchor_grid_y_down_40,
            ),
            axis=0,
        )

    @property
    def anchor_grid_2d(self):
        anchor_grid_2d = np.concatenate(
            [
                self.anchor_grid_x[:, :, np.newaxis],
                self.anchor_grid_y[:, :, np.newaxis],
            ],
            axis=-1,
        )
        return anchor_grid_2d

    @property
    def anchor_grid_x(self):
        return np.repeat(
            np.expand_dims(self.anchor_x_steps, axis=0), self.anchor_num, axis=0
        )

    @property
    def anchor_grid_3d_flat(self):
        anchor_grid_2d = self.anchor_grid_2d
        anchor_grid_3d = np.zeros((self.anchor_num, self.num_x_steps, 3))
        anchor_grid_3d[:, :, :2] = anchor_grid_2d
        return anchor_grid_3d
