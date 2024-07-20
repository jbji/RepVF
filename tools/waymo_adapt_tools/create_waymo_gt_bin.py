try:
    from waymo_open_dataset import dataset_pb2, label_pb2
    from waymo_open_dataset.protos import metrics_pb2
except ImportError:
    raise ImportError(
        'Please run "pip3 install waymo-open-dataset-tf-2-11-0==1.5.1" "pip3 install Pillow==9.2.0" "pip install tensorflow==2.11.0" '
        "to install the ungraded official devkit first. Also we recommend using waymo 1.4.2 dataset"
    )
import argparse
import os
from os.path import join
import re
import numpy as np
from openlanev1.io import io
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Waymo GT Generator arg parser")
parser.add_argument(
    "--pkl-path",
    type=str,
    default="data/waymo/openlane_format/validation.pkl",
    help="specify the collected labels of the validation part",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="data/waymo/openlane_format/",
    required=False,
    help="where the generated gt.bin will be saved",
)
parser.add_argument(
    "--bin-name",
    type=str,
    default="cam_gt.bin",
    required=False,
    help="name of the generated gt.bin",
)
args = parser.parse_args()


def match_context_name(text):
    match = re.search(r"segment-(\d+_\d+_\d+_\d+_\d+)", text)
    if match:
        extracted_string = match.group(1)
        return extracted_string
    return ""


def generate_gt_bin():
    gt_pkl = io.pickle_load(args.pkl_path)
    objects = metrics_pb2.Objects()
    label_dict = {
        "VEHICLE": label_pb2.Label.TYPE_VEHICLE,
        "PEDESTRIAN": label_pb2.Label.TYPE_PEDESTRIAN,
        "CYCLIST": label_pb2.Label.TYPE_CYCLIST,
        "SIGN": label_pb2.Label.TYPE_SIGN,
    }
    for i, frame_id in tqdm(enumerate(gt_pkl), desc="Generating cam_gt.bin"):
        # dict_keys(['intrinsics', 'extrinsics', 'pose', 'path_point_cloud', 'path_cameras', 'lane_lines', 'gt_boxes', 'gt_names', 'gt_velocity'])
        # bboxes with no point cloud points or no camera can seen are filtered out during convert/collect step
        # more specifically,
        # 1 in conversion step, bboxes with lidar points less than 1 are filtered out
        # 2 in collection step, bboxes with unknown camera are filtered out
        for index, (bbox, name) in enumerate(
            zip(gt_pkl[frame_id]["gt_boxes"], gt_pkl[frame_id]["gt_names"])
        ):
            box3d = label_pb2.Label.Box()
            box3d.center_x = bbox[0]
            box3d.center_y = bbox[1]
            box3d.center_z = bbox[2]
            box3d.length = bbox[3]
            box3d.width = bbox[4]
            box3d.height = bbox[5]

            heading = bbox[6]
            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi
            box3d.heading = heading

            o = metrics_pb2.Object()
            o.context_name = match_context_name(frame_id[0])
            o.frame_timestamp_micros = int(frame_id[1][:-2])
            o.score = 0.5
            o.object.num_lidar_points_in_box = 5
            o.object.num_top_lidar_points_in_box = 5
            # o.object.id = index

            if name.upper() in label_dict:
                o.object.type = label_dict[name.upper()]
            else:
                raise ValueError("Unknown label type")

            o.object.box.CopyFrom(box3d)

            objects.objects.append(o)

    out_dir = args.out_dir
    f = open(join(out_dir, args.bin_name), "wb")
    f.write(objects.SerializeToString())
    f.close()


if __name__ == "__main__":
    generate_gt_bin()
