# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip3 install waymo-open-dataset-tf-2-11-0==1.5.1" "pip3 install Pillow==9.2.0" "pip install tensorflow==2.11.0" '
        'to install the ungraded official devkit first. Also we recommend using waymo 1.4.2 dataset')

from glob import glob
from os.path import join

import mmcv
import numpy as np
from openlanev1.io import io
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import \
    parse_range_image_and_camera_projection


class Waymo2Waymo(object):
    """Unpack Waymo Dataset

    This class serves as the converter to change the waymo raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 split,
                 prefix,
                 workers=64,
                 test_mode=False,
                 verbose=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = [
            '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
            '_SIDE_LEFT'
        ]
        self.type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }

        self.load_dir = join(load_dir, split)
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode
        self.split = split
        self.verbose = verbose

        self.tfrecord_pathnames = sorted(
            glob(join(self.load_dir, '*.tfrecord')))

        # self.label_save_dir = f'{self.save_dir}/label_'
        # self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/images_'
        # self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne/{split}'
        self.info_save_dir = f'{self.save_dir}/detection3d_1000/{split}'
        # self.pose_save_dir = f'{self.save_dir}/pose'
        # self.timestamp_save_dir = f'{self.save_dir}/timestamp'

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                     self.workers)
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        segment_str = pathname.split('/')[-1].split('.')[0]
        # print(f'Converting {pathname} ...')
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')
        self.create_subfolders(segment_str)
        if self.verbose:
            print(f'Working on {pathname} ...')
        for frame_idx, data in enumerate(dataset):
            if self.verbose:
                print(f'Working on {pathname} frame {frame_idx} ...')
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (self.selected_waymo_locations is not None
                    and frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue
            
            timestamp_str = str(frame.timestamp_micros) + '00'
            self.save_image(frame, segment_str, timestamp_str)
            self.save_lidar(frame, segment_str, timestamp_str)
            
            # calib, pose and label
            pose = self.get_pose(frame, segment_str, timestamp_str)
            calib = self.get_calib(frame, segment_str, timestamp_str)
            file_paths = {
                'path_point_cloud': f'velodyne/{self.split}/{segment_str}/{timestamp_str}.bin',
                'path_cameras': [f'images_{str(img.name - 1)}/{self.split}/{segment_str}/{timestamp_str}.jpg' for img in frame.images]
            }
            if not self.test_mode:
                label = self.get_label(frame)
                self.save_info([calib, pose, label, file_paths], segment_str, timestamp_str)
            else:
                self.save_info([calib, pose, file_paths], segment_str, timestamp_str)

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, segment_str, timestamp_str):
        """Parse and save the images in jpg format. Jpg is the original format
        used by Waymo Open dataset. Saving in png format will cause huge (~3x)
        unnesssary storage waste.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            segment_str (str): Segment string.
        """
        for img in frame.images:
            img_path = f'{self.image_save_dir}{str(img.name - 1)}/{self.split}/' + segment_str + '/'+ \
                f'{timestamp_str}.jpg'
            with open(img_path, 'wb') as fp:
                fp.write(img.image)

    def save_info(self, info_list, segment_str, timestamp_str):
        json_dict = {}
        [json_dict.update(info) for info in info_list]
        io.json_save(json_dict, f'{self.info_save_dir}/{segment_str}/' + timestamp_str + '.json')
    
    def get_calib(self, frame, segment_str, timestamp_str):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        
        camera_extrinsics = []
        camera_intrinsics = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            camera_extrinsic = np.array(camera.extrinsic.transform)
            camera_extrinsic.resize((4, 4))
            camera_extrinsic = camera_extrinsic[:3, :]
            # convert np.array to list for json saving
            camera_extrinsic = camera_extrinsic.tolist()
            camera_extrinsics.append(camera_extrinsic)

            # intrinsic parameters
            camera_calib = np.zeros((3, 3))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = camera_calib.tolist()
            camera_intrinsics.append(camera_calib)

        json_dict = {
            'intrinsics': camera_intrinsics, 
            'extrinsics': camera_extrinsics
            }
        
        return json_dict
        

    def save_lidar(self, frame, segment_str, timestamp_str):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, _, range_image_top_pose = \
            parse_range_image_and_camera_projection(frame)

        # First return
        points_0, cp_points_0, intensity_0, elongation_0, mask_indices_0 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)
        mask_indices_0 = np.concatenate(mask_indices_0, axis=0)

        # Second return
        points_1, cp_points_1, intensity_1, elongation_1, mask_indices_1 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)
        mask_indices_1 = np.concatenate(mask_indices_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        mask_indices = np.concatenate([mask_indices_0, mask_indices_1], axis=0)

        # timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
        point_cloud = np.column_stack(
            (points, intensity, elongation, mask_indices))

        pc_path = f'{self.point_cloud_save_dir}/{segment_str}/{self.prefix}' + \
            f'{timestamp_str}.bin'
        point_cloud.astype(np.float32).tofile(pc_path)


    def get_label(self, frame):
        label_data = []
        id_to_bbox = dict()
        id_to_name = dict()

        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            camera = None
            id = obj.id

            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    camera = lidar
                    break

            if bounding_box is None or name is None or camera is None:
                name = '0'
                camera = 'UNKNOWN'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue
            
            # Don't convert to KITTI Class
            # my_type = self.waymo_to_kitti_class_map[my_type]

            height = obj.box.height
            width = obj.box.width
            length = obj.box.length
            velocity_x = obj.metadata.speed_x
            velocity_y = obj.metadata.speed_y

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z

            label_info = {
                'type': my_type,
                'bbox_2d': bounding_box,
                'bbox_2d_camera': camera,
                'dimensions': [height, width, length],
                'center_location': [x, y, z],
                'rotation_y': obj.box.heading,
                'velocity': [velocity_x, velocity_y],
            }

            if self.save_track_id:
                label_info['name'] = name
                label_info['track_id'] = obj.id

            label_data.append(label_info)

        return {'bbox': label_data}


    def save_label(self, frame, file_idx, frame_idx):
        """Parse and save the label data in txt format.
        The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        fp_label_all = open(
            f'{self.label_all_save_dir}/{self.prefix}' +
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'w+')
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            if bounding_box is None or name is None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            my_type = self.waymo_to_kitti_class_map[my_type]

            height = obj.box.height
            width = obj.box.width
            length = obj.box.length

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2

            # project bounding box to the virtual reference frame
            pt_ref = self.T_velo_to_front_cam @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -obj.box.heading - np.pi / 2
            track_id = obj.id

            # not available
            truncated = 0
            occluded = 0
            alpha = -10

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            if self.save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            fp_label = open(
                f'{self.label_save_dir}{name}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def get_pose(self, frame, file_idx, frame_idx):
        """Parse and return the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        return {'pose': pose.tolist()}
        # np.savetxt(
        #     join(f'{self.pose_save_dir}/{self.prefix}' +
        #          f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
        #     pose)

    def save_timestamp(self, frame, file_idx, frame_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        with open(
                join(f'{self.timestamp_save_dir}/{self.prefix}' +
                     f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
                'w') as f:
            f.write(str(frame.timestamp_micros))

    def create_subfolders(self, segment_name):
        if not self.test_mode:
            dir_list1 = [
                # self.label_all_save_dir, 
                self.point_cloud_save_dir,
                self.info_save_dir,
            ]
            dir_list2 = [self.image_save_dir]
        else:
            dir_list1 = [
                self.point_cloud_save_dir,
                self.info_save_dir,
            ]
            dir_list2 = [self.image_save_dir]
        
        for d in dir_list1:
            mmcv.mkdir_or_exist(join(d, segment_name))
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(join(f'{d}{str(i)}/{self.split}', segment_name))

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list1 = [
                # self.label_all_save_dir, 
                self.point_cloud_save_dir, 
                self.info_save_dir,
            ]
            dir_list2 = [self.image_save_dir]
        else:
            dir_list1 = [
                self.point_cloud_save_dir,
                self.info_save_dir,
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(f'{d}{str(i)}/{self.split}')

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.

        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.

        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], intensity
                with shape [N, 1], elongation with shape [N, 1], points'
                position in the depth map (element offset if points come from
                the main lidar otherwise -1) with shape[N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []
        elongation = []
        mask_indices = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            mask_index = tf.where(range_image_mask)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)
            elongation.append(elongation_tensor.numpy())
            if c.name == 1:
                mask_index = (ri_index * range_image_mask.shape[0] +
                              mask_index[:, 0]
                              ) * range_image_mask.shape[1] + mask_index[:, 1]
                mask_index = mask_index.numpy().astype(elongation[-1].dtype)
            else:
                mask_index = np.full_like(elongation[-1], -1)

            mask_indices.append(mask_index)

        return points, cp_points, intensity, elongation, mask_indices

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret
