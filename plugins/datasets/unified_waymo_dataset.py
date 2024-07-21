try:
    from waymo_open_dataset import dataset_pb2, label_pb2
    from waymo_open_dataset.protos import metrics_pb2
except ImportError:
    raise ImportError(
        'Please run "pip3 install waymo-open-dataset-tf-2-11-0==1.5.1" "pip3 install Pillow==9.2.0" "pip install tensorflow==2.11.0" '
        "to install the ungraded official devkit first. Also we recommend using waymo 1.4.2 dataset"
    )
import json
import os
import re
import tempfile
from os import makedirs
from os.path import exists, expanduser, join, split
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyquaternion
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import Custom3DDataset
from mmdet.datasets import DATASETS
from nuscenes.utils.data_classes import Box as NuScenesBox
from openlanev1.dataset.multiview import MultiViewCollection
from openlanev1.evaluation import LaneEval
from openlanev1.utils import define_args
from tqdm import tqdm

import mmcv


@DATASETS.register_module()
class UnifiedWaymoDataset(Custom3DDataset):
    CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]

    def __init__(
        self,
        data_root,
        collection,
        pipeline,
        test_mode=False,
        box_type_3d="LiDAR",
        classes=None,
        filter_empty_gt=True,
        with_velocity=True,
        bbox_metric=None,  # support mAP or LET_mAP.
        bin_name="cam_gt.bin",
        det_eval_kit_root="mmdetection3d/projects/uni/tools/waymo_utils/",
    ):
        self.ann_file = f"{join(expanduser(data_root),collection)}.pkl"
        self.with_velocity = with_velocity
        self.metric = bbox_metric
        self.waymo_bin_name = bin_name
        self.eval_kit_root = det_eval_kit_root
        super().__init__(
            data_root=data_root,
            ann_file=self.ann_file,
            pipeline=pipeline,
            classes=classes,
            test_mode=test_mode,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
        )

    def load_annotations(self, ann_file):
        ann_file = ann_file.name.split(".pkl")[0].split("/")
        self.collection = MultiViewCollection(
            data_root=self.data_root,
            collection=ann_file[-1],
        )
        return self.collection.keys

    def get_data_info(self, index):
        segment_id, timestamp = self.data_infos[index]
        frame = self.collection.get_frame_via_identifier((segment_id, timestamp))

        input_dict = {
            "scene_token": segment_id,
            "sample_idx": timestamp,
            "img_paths": frame.get_image_paths(),
            "calib": {
                "intrinsics": frame.get_intrinsics(),
                "extrinsics": frame.get_extrinsics(),
            },
        }

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos
            input_dict["bbox3d_fields"] = []
        return input_dict

    def get_ann_info(self, index):
        segment_id, timestamp = self.data_infos[index]
        frame = self.collection.get_frame_via_identifier((segment_id, timestamp))

        lane_lines = frame.get_annotations_lane_lines()

        # Prepare gt_lane_lines
        gt_lane_lines = []
        for lane_line in lane_lines:
            gt_lane_line = {
                "category": lane_line["category"],
                "visibility": lane_line["visibility"],
                "uv": lane_line["uv"],
                "xyz": lane_line["xyz"],
            }
            gt_lane_lines.append(gt_lane_line)

        # Prepare gt_bbox3d
        gt_bboxes_3d = frame.get_annotations_bbox3d()
        gt_names_3d = frame.get_annotations_bbox3d_names()
        gt_labels_3d = []

        # only for debugging
        # gt_bboxes_3d = []
        # gt_names_3d = []

        assert (
            len(gt_bboxes_3d) > 0 or len(gt_lane_lines) > 0
        ), f"no gt bbox3d or lane line in {segment_id}/{timestamp}, please filter out this frame"

        assert (
            gt_bboxes_3d.shape[1] == 7
        ), f"gt bbox3d shape is not correct, please check {segment_id}/{timestamp}"

        # print(f"in dataset get_item, gt shape: {gt_bboxes_3d.shape}")
        if len(gt_bboxes_3d) > 0:
            for cat in gt_names_3d:
                if cat in self.CLASSES:
                    gt_labels_3d.append(self.CLASSES.index(cat))
                else:
                    gt_labels_3d.append(-1)

            if self.with_velocity:
                gt_velocity = frame.get_annotations_bbox3d_velocity()
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
                # print(gt_bboxes_3d, gt_velocity)
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

            gt_names_3d = np.array(gt_names_3d)
            gt_labels_3d = np.array(gt_labels_3d, dtype=np.long)
            # gt_bboxes_3d is a list of arraies, each np array is a bbox3d, x,y,z,l,w,h,yaw
            gt_bboxes_3d = LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
            ).convert_to(self.box_mode_3d)
        else:
            gt_names_3d = np.array(gt_names_3d)
            gt_labels_3d = np.array(gt_labels_3d, dtype=np.long)
            gt_bboxes_3d = LiDARInstance3DBoxes(
                np.zeros((0, 7)), box_dim=7, origin=(0.5, 0.5, 0.5)
            ).convert_to(self.box_mode_3d)

        return {
            "gt_lane_lines": gt_lane_lines,
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "gt_names": gt_names_3d,
        }

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        # only for debugging
        # print(
        #     f"after data pipeline, {example['img_metas'].data['scene_token']}'s gt bbox shape: {len(example['gt_bboxes_3d'].data)},{example['gt_bboxes_3d'].data[0].tensor.shape if len(example['gt_bboxes_3d'].data) > 0 else None}"
        # )
        # fallback
        if "gt_bboxes_3d" in example.keys() and len(example["gt_bboxes_3d"].data) == 0:
            return None

        if (
            "gt_lanes_3d" in example.keys()
            and len(example["gt_lanes_3d"]["lane_pts"]) == 0
        ):
            return None
        return example

    def _format_3dlane(self, results):
        json_list = []
        for result_list in tqdm(results, desc="formatting predictions"):
            lane3d_results_dict = result_list["lane3d"]
            json_list.append(lane3d_results_dict)
        return json_list

    def _evaluate_3dlane(
        self,
        pred_dict,
        logger,
        **kwargs,
    ):
        if os.environ.get("SAVE_LANE3D_PREDS") == "True":
            output_dir = os.environ.get(
                "OUTPUT_DIR", "./visualizations/"
            )  # Default to current directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_name = f"all_lane3d_preds.npy"  # Default filename
            output_path = os.path.join(output_dir, file_name)
            np.save(output_path, pred_dict)

        evaluator = LaneEval()
        eval_stats = evaluator.bench_one_submit_runtime(
            ground_truth=self.ann_file, predictions=pred_dict, prob_th=0.5
        )

        if logger is not None:
            logger.info(
                "===> Evaluation on validation set: \n"
                "laneline F-measure {:.8} \n"
                "laneline Recall  {:.8} \n"
                "laneline Precision  {:.8} \n"
                "laneline Category Accuracy  {:.8} \n"
                "laneline x error (close)  {:.8} m\n"
                "laneline x error (far)  {:.8} m\n"
                "laneline z error (close)  {:.8} m\n"
                "laneline z error (far)  {:.8} m\n".format(
                    eval_stats[0],
                    eval_stats[1],
                    eval_stats[2],
                    eval_stats[3],
                    eval_stats[4],
                    eval_stats[5],
                    eval_stats[6],
                    eval_stats[7],
                )
            )

        metric_results = {
            "laneline F-measure": eval_stats[0],
            "laneline Recall": eval_stats[1],
            "laneline Precision": eval_stats[2],
            "laneline Category Accuracy": eval_stats[3],
            "laneline x error (close)": eval_stats[4],
            "laneline x error (far)": eval_stats[5],
            "laneline z error (close)": eval_stats[6],
            "laneline z error (far)": eval_stats[7],
        }
        return metric_results

    def _format_3dbbox_waymo(self, results, binfile_name):
        # first format the results by putting them together, or consider convert to kitti format
        # results[0].keys()
        # dict_keys(['pts_bbox', 'lane3d'])
        # results[0]['pts_bbox'].keys()
        # dict_keys(['boxes_3d', 'scores_3d', 'labels_3d'])
        # results[0]['pts_bbox']['boxes_3d'].tensor.shape torch.Size([256, 9])
        # results[0]['pts_bbox']['scores_3d'].shape 256
        for i, res in enumerate(results):
            res["pts_bbox"]["boxes_3d"].limit_yaw(offset=0.5, period=np.pi * 2)
        # then convert predtion to waymo bin file so that we could compute metric.
        objects = metrics_pb2.Objects()
        k2w_cls_map = {
            "Car": label_pb2.Label.TYPE_VEHICLE,
            "Vehicle": label_pb2.Label.TYPE_VEHICLE,
            "Pedestrian": label_pb2.Label.TYPE_PEDESTRIAN,
            "Sign": label_pb2.Label.TYPE_SIGN,
            "Cyclist": label_pb2.Label.TYPE_CYCLIST,
        }

        def match_context_name(text):
            match = re.search(r"segment-(\d+_\d+_\d+_\d+_\d+)", text)
            if match:
                extracted_string = match.group(1)
                return extracted_string
            return ""

        for res in results:
            lidar_boxes = res["pts_bbox"]["boxes_3d"].tensor
            scores = res["pts_bbox"]["scores_3d"]
            labels = res["pts_bbox"]["labels_3d"]
            timestamp = int(res["sample_idx"][:-2])
            for lidar_box, score, label in zip(lidar_boxes, scores, labels):
                box = label_pb2.Label.Box()
                height = lidar_box[5].item()
                heading = lidar_box[6].item()
                while heading < -np.pi:
                    heading += 2 * np.pi
                while heading > np.pi:
                    heading -= 2 * np.pi
                box.center_x = lidar_box[0].item()
                box.center_y = lidar_box[1].item()
                box.center_z = lidar_box[2].item() + height / 2
                box.length = lidar_box[3].item()
                box.width = lidar_box[4].item()
                box.height = height
                box.heading = heading

                o = metrics_pb2.Object()
                o.object.box.CopyFrom(box)
                o.context_name = match_context_name(res["scene_token"])
                o.object.type = k2w_cls_map[self.CLASSES[label]]
                o.object.num_lidar_points_in_box = 5
                # o.object.num_top_lidar_points_in_box = 5
                o.score = score.item()
                o.frame_timestamp_micros = timestamp

                objects.objects.append(o)

        out_dir = tempfile.TemporaryDirectory().name
        if not exists(out_dir):
            makedirs(out_dir)
        f = open(join(out_dir, f"{binfile_name}.bin"), "wb")
        f.write(objects.SerializeToString())
        f.close()

        return results, out_dir

    def _print_log(self, text, logger):
        if logger is not None:
            logger.info(text)

    def _evaluate_waymo(
        self,
        pklfile_prefix: str,
        logger=None,
    ) -> Dict[str, float]:
        """Evaluation in Waymo protocol.
        This function is based on WaymoMetric from mmdet3d 1.2.0.

        Args:
            pklfile_prefix (str): The location that stored the prediction
                results.
            metric (str, optional): Metric to be evaluated. Defaults to None.
            use mAP or LET_mAP.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """

        import subprocess

        waymo_bin_file = join(self.data_root, self.waymo_bin_name)

        if self.metric == "mAP":
            eval_kit = join(self.eval_kit_root, "compute_detection_metrics_main")
            eval_str = eval_kit + f" {pklfile_prefix}.bin " + f"{waymo_bin_file}"
            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode("utf-8")
            self._print_log(ret_texts, logger=logger)

            ap_dict = {
                "Vehicle/L1 mAP": 0,
                "Vehicle/L1 mAPH": 0,
                "Vehicle/L2 mAP": 0,
                "Vehicle/L2 mAPH": 0,
                "Pedestrian/L1 mAP": 0,
                "Pedestrian/L1 mAPH": 0,
                "Pedestrian/L2 mAP": 0,
                "Pedestrian/L2 mAPH": 0,
                "Sign/L1 mAP": 0,
                "Sign/L1 mAPH": 0,
                "Sign/L2 mAP": 0,
                "Sign/L2 mAPH": 0,
                "Cyclist/L1 mAP": 0,
                "Cyclist/L1 mAPH": 0,
                "Cyclist/L2 mAP": 0,
                "Cyclist/L2 mAPH": 0,
                "Overall/L1 mAP": 0,
                "Overall/L1 mAPH": 0,
                "Overall/L2 mAP": 0,
                "Overall/L2 mAPH": 0,
            }
            mAP_splits = ret_texts.split("mAP ")
            mAPH_splits = ret_texts.split("mAPH ")
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 2) + 1
                if idx % 2 == 0:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split("]")[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split("]")[0])
            ap_dict["Overall/L1 mAP"] = (
                ap_dict["Vehicle/L1 mAP"]
                + ap_dict["Pedestrian/L1 mAP"]
                + ap_dict["Cyclist/L1 mAP"]
            ) / 3
            ap_dict["Overall/L1 mAPH"] = (
                ap_dict["Vehicle/L1 mAPH"]
                + ap_dict["Pedestrian/L1 mAPH"]
                + ap_dict["Cyclist/L1 mAPH"]
            ) / 3
            ap_dict["Overall/L2 mAP"] = (
                ap_dict["Vehicle/L2 mAP"]
                + ap_dict["Pedestrian/L2 mAP"]
                + ap_dict["Cyclist/L2 mAP"]
            ) / 3
            ap_dict["Overall/L2 mAPH"] = (
                ap_dict["Vehicle/L2 mAPH"]
                + ap_dict["Pedestrian/L2 mAPH"]
                + ap_dict["Cyclist/L2 mAPH"]
            ) / 3
        elif self.metric == "LET_mAP":
            eval_kit = join(self.eval_kit_root, "compute_detection_let_metrics_main")
            eval_str = (
                eval_kit
                + f"compute_detection_let_metrics_main {pklfile_prefix}.bin "
                + f"{waymo_bin_file}"
            )

            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode("utf-8")

            self._print_log(ret_texts, logger=logger)
            ap_dict = {
                "Vehicle mAPL": 0,
                "Vehicle mAP": 0,
                "Vehicle mAPH": 0,
                "Pedestrian mAPL": 0,
                "Pedestrian mAP": 0,
                "Pedestrian mAPH": 0,
                "Sign mAPL": 0,
                "Sign mAP": 0,
                "Sign mAPH": 0,
                "Cyclist mAPL": 0,
                "Cyclist mAP": 0,
                "Cyclist mAPH": 0,
                "Overall mAPL": 0,
                "Overall mAP": 0,
                "Overall mAPH": 0,
            }
            mAPL_splits = ret_texts.split("mAPL ")
            mAP_splits = ret_texts.split("mAP ")
            mAPH_splits = ret_texts.split("mAPH ")
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 3) + 1
                if idx % 3 == 0:  # mAPL
                    ap_dict[key] = float(mAPL_splits[split_idx].split("]")[0])
                elif idx % 3 == 1:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split("]")[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split("]")[0])
            ap_dict["Overall mAPL"] = (
                ap_dict["Vehicle mAPL"]
                + ap_dict["Pedestrian mAPL"]
                + ap_dict["Cyclist mAPL"]
            ) / 3
            ap_dict["Overall mAP"] = (
                ap_dict["Vehicle mAP"]
                + ap_dict["Pedestrian mAP"]
                + ap_dict["Cyclist mAP"]
            ) / 3
            ap_dict["Overall mAPH"] = (
                ap_dict["Vehicle mAPH"]
                + ap_dict["Pedestrian mAPH"]
                + ap_dict["Cyclist mAPH"]
            ) / 3
        return ap_dict

    def evaluate(
        self,
        results,
        logger=None,
        # for 3d bbox evaluation
        show=False,
        out_dir=None,
        result_names=["pts_bbox"],
        # for 3d lane evaluation
        dump=None,
        dump_dir=None,
        visualization=False,
        visualization_dir=None,
        visualization_num=None,
        **kwargs,
    ):
        # format results
        self._print_log(f"Formating results...", logger=logger)
        binfile_name = "results"

        has_bbox3d = "pts_bbox" in results[0].keys()
        if has_bbox3d:
            pred_3dbbox_waymo, tmp_dir = self._format_3dbbox_waymo(
                results=results, binfile_name=binfile_name
            )

        has_lane3d = "lane3d" in results[0].keys()
        if has_lane3d:
            pred_dict_3dlane = self._format_3dlane(results)

        metric_dict = {}
        if has_bbox3d and os.environ.get("SAVE_LANE3D_PREDS") != "True":
            self._print_log(f"Evaluating {self.metric}...", logger=logger)
            metric_bbox_ap = self._evaluate_waymo(
                pklfile_prefix=join(tmp_dir, binfile_name), logger=logger
            )
            metric_dict.update(metric_bbox_ap)

        if has_lane3d:
            self._print_log(f"Evaluating 3d lane...", logger=logger)
            metric_3dlane = self._evaluate_3dlane(pred_dict_3dlane, logger)
            metric_dict.update(metric_3dlane)

        return metric_dict


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"
):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list
