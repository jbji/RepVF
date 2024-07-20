# =========================================================
# author: jbji
# date: 2023-03-26
# =========================================================

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from os.path import expanduser, join

import numpy as np
from openlanev1.io import io
from tqdm import tqdm


def _parse_label_key(path):
    path_elems = path.split("/")
    split = path_elems[-3]
    segement_id = path_elems[-2]
    frame_id = path_elems[-1].split(".")[0]
    return split, segement_id, frame_id


def _parse_label_keys(path_list):
    data_list = []
    for path in path_list:
        data_list.append(_parse_label_key(path))
    return data_list


def collect(
    data_root: str, meta_root: str, collection: str, point_interval: int = 1
) -> None:
    r"""
    Load meta data of collection for openlanev1 dataset
    and store in a .pkl with split as file name.

    Parameters
    ----------
    data_root : str
        Path of data root. (images) e.g. data/openlanev1/images
    meta_root : str
        Path of meta root. (annotations) e.g. data/openlanev1/lane3d_300
    point_interval : int
        Interval for subsampling points of lane centerlines,
        not subsampling as default.

    """
    data_root = expanduser(data_root)
    meta_root = expanduser(meta_root)

    label_list = glob(
        join(meta_root, collection, "segment-*", "*.json"), recursive=True
    )
    if len(label_list) == 0:
        raise FileNotFoundError(
            "We can't find any label file in the collection. Please check the path of meta_root and collection."
        )

    data_list = _parse_label_keys(label_list)

    # dict_keys(['extrinsic', 'intrinsic', 'file_path', 'lane_lines'])
    # one sensor, one frame, one lane, each frame is a key frame.
    # extrinsic: 4x3 matrix, rotation and translation
    # intrinsic: 3x3 matrix, camera intrinsic K

    meta = {}
    for split, segment_id, timestamp in tqdm(data_list, desc="Collecting meta data"):
        raw_json = io.json_load(f"{meta_root}/{split}/{segment_id}/{timestamp}.json")
        raw_json["version"] = "OpenLaneV1"
        raw_json["segment_id"] = segment_id
        raw_json["meta_data"] = {
            "source": "waymo",
            "source_seg_id": segment_id,
            "source_frame_id": timestamp,
        }
        raw_json["timestamp"] = timestamp

        meta[(split, segment_id, timestamp)] = raw_json

    skip_cnt = 0
    # convert matrix into numpy array
    for identifier, frame in tqdm(
        meta.items(), desc="Converting matrix to numpy array"
    ):
        if "cam_pitch" in frame:
            print("cam_pitch is deprecated, please use extrinsic instead.")
        # meta_single['extrinsic'] = {
        #     'rotation': np.array(frame['extrinsic'][0:3], dtype=np.float64),
        #     'translation': np.array(frame['extrinsic'][3], dtype=np.float64)
        # }
        # meta_single['extrinsic'] = np.array(frame['extrinsic'], dtype=np.float64)
        meta[identifier]["extrinsic"] = {
            "rot_trans": np.array(frame["extrinsic"], dtype=np.float64)
        }
        meta[identifier]["intrinsic"] = {
            "k": np.array(frame["intrinsic"], dtype=np.float64)
        }
        # subsample points
        if "lane_lines" not in frame:
            skip_cnt += 1
            continue
        # category, visibility, uv, xyz
        for i, lane_line in enumerate(frame["lane_lines"]):
            meta[identifier]["lane_lines"][i]["visibility"] = np.array(
                lane_line["visibility"][::point_interval], dtype=np.float32
            )
            meta[identifier]["lane_lines"][i]["uv"] = np.array(
                [uv[::point_interval] for uv in lane_line["uv"]], dtype=np.float32
            )
            meta[identifier]["lane_lines"][i]["xyz"] = np.array(
                [xyz[::point_interval] for xyz in lane_line["xyz"]], dtype=np.float32
            )

    print(f"Skipped {skip_cnt} frames without lanelines.")

    io.pickle_dump(f"{meta_root}/{collection}.pkl", meta)


root_dict_example = {
    "openlane_format": "~/data/waymo/1_4_2/openlane_format",
    "detection3d": "detection3d_1000",
    "lane3d": "lane3d_300",
}

split_config_example = {
    "suffix": None,
    "method": "detection3d",  # 'detection3d', 'lane3d', 'random'
}

split_config_example_random = {
    "suffix": None,
    "method": "random",
    "ratio": "min",  # 'min', 'max', 'avg'
    "seed": 42,
}


def _parse_label_key_multiview(path):
    path_elems = path.split("/")
    segement_id = path_elems[-2]
    time_stamp = path_elems[-1].split(".")[0]
    return segement_id, time_stamp


def _parse_label_keys_multiview(path_list: list) -> list:
    data_list = []
    for path in path_list:
        data_list.append(_parse_label_key_multiview(path))
    return data_list


def _dump_process_identifier(identifier, mapping, root_dict):
    meta_single = {}
    segment_id, timestamp = identifier
    for source in ["detection3d", "lane3d"]:
        source_split = mapping[identifier][source]
        path_json = join(
            root_dict["openlane_format"],
            root_dict[source],
            source_split,
            segment_id,
            timestamp + ".json",
        )
        meta_single.update(io.json_load(path_json))

    # remove redundant keys
    meta_single["path_cameras"].sort()
    for key_name in ["extrinsic", "intrinsic", "file_path"]:
        if key_name in meta_single:
            meta_single.pop(key_name)

    # for each item replace with corresponding np array,
    for i, intrinsic in enumerate(meta_single["intrinsics"]):
        meta_single["intrinsics"][i] = np.array(intrinsic, dtype=np.float64)
    for i, extrinsic in enumerate(meta_single["extrinsics"]):
        meta_single["extrinsics"][i] = np.array(extrinsic, dtype=np.float64)
    for i, lane_line in enumerate(meta_single["lane_lines"]):
        meta_single["lane_lines"][i]["visibility"] = np.array(
            lane_line["visibility"], dtype=np.float32
        )
        meta_single["lane_lines"][i]["uv"] = np.array(
            [uv for uv in lane_line["uv"]], dtype=np.float32
        )
        meta_single["lane_lines"][i]["xyz"] = np.array(
            [xyz for xyz in lane_line["xyz"]], dtype=np.float32
        )

    # convert bbox to gt_boxes and gt_names
    gt_boxes = []
    gt_names = []
    gt_velocity = []
    for i, bbox in enumerate(meta_single["bbox"]):
        if bbox["bbox_2d_camera"] == "UNKNOWN":
            continue
        h, w, l = bbox["dimensions"]
        rot_z = bbox["rotation_y"]
        gt_boxes.append(
            np.concatenate(
                [bbox["center_location"], [l, w, h], [rot_z]], axis=None
            ).astype(np.float32)
        )
        gt_names.append(bbox["type"].capitalize())
        gt_velocity.append(bbox["velocity"])
    meta_single.pop("bbox")
    meta_single["gt_boxes"] = np.array(gt_boxes, dtype=np.float32)
    meta_single["gt_names"] = np.array(gt_names, dtype=np.str)
    meta_single["gt_velocity"] = np.array(gt_velocity, dtype=np.float32)
    return meta_single


def _dump_helper_multiview(
    labels: list, mapping: dict, root_dict: dict, max_workers=16
) -> dict:
    meta = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _dump_process_identifier, identifier, mapping, root_dict
            ): identifier
            for identifier in labels
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing meta data"
        ):
            identifier = futures[future]
            try:
                meta[identifier] = future.result()
            except Exception as exc:
                print(f"{identifier} generated an exception: {exc}")
    return meta


def _dump_helper_multiview_single_thread(
    labels: list, mapping: dict, root_dict: dict
) -> dict:
    meta = {}
    for segment_id, timestamp in tqdm(labels, desc="Processing meta data"):
        identifier = (segment_id, timestamp)
        meta[identifier] = _dump_process_identifier(identifier, mapping, root_dict)
    return meta


def collect_multiview(
    root_dict: dict,
    split_config: dict,
    to_dump: str = "both",  # 'both', 'train', 'val'
    max_workers: int = 16,
    filter_empty_gt: bool = True,
) -> None:
    root_dict["openlane_format"] = expanduser(root_dict["openlane_format"])
    openlane_root = root_dict["openlane_format"]

    # Load label list and create split mapping
    print("Loading label lists...")
    label_lists = []
    split_mapping = {}
    for source in ["detection3d", "lane3d"]:
        for split in ["training", "validation"]:
            labels = glob(
                join(openlane_root, root_dict[source], split, "segment-*", "*.json"),
                recursive=True,
            )
            labels = _parse_label_keys_multiview(labels)
            label_lists.append(labels)
            for segment_id, timestamp in labels:
                identifier = (segment_id, timestamp)
                if identifier not in split_mapping:
                    split_mapping[identifier] = {}
                split_mapping[identifier][source] = split  # Map label to split

    # filter out frames that don't have both labels
    filtered_label_lists = []
    for labels in tqdm(label_lists, desc="Filtering labels for all splits"):
        filtered_labels = []
        for segment_id, timestamp in tqdm(
            labels, desc="Filtering labels for one split"
        ):
            identifier = (segment_id, timestamp)
            if identifier in split_mapping and len(split_mapping[identifier]) == 2:
                filtered_labels.append(identifier)
        filtered_label_lists.append(filtered_labels)

    # Process data split
    print("Processing data split...")
    train_labels, val_labels = [], []
    if split_config["method"] == "detection3d":
        train_labels, val_labels = (
            filtered_label_lists[0],
            filtered_label_lists[1],
        )  # Use detection3d split
    elif split_config["method"] == "lane3d":
        train_labels, val_labels = (
            filtered_label_lists[2],
            filtered_label_lists[3],
        )  # Use lane3d split
    elif split_config["method"] == "random":
        seed = split_config["seed"]
        random.seed(seed)
        # Merge all data
        all_labels = split_mapping.keys()
        # Decide split ratio
        split_ratio = 0.8
        if split_config["ratio"] == "min":
            split_ratio = min(
                len(filtered_label_lists[0]), len(filtered_label_lists[2])
            ) / len(all_labels)
        elif split_config["ratio"] == "max":
            split_ratio = max(
                len(filtered_label_lists[0]), len(filtered_label_lists[2])
            ) / len(all_labels)
        elif split_config["ratio"] == "average":
            split_ratio = (
                len(filtered_label_lists[0]) + len(filtered_label_lists[2])
            ) / (2 * len(all_labels))
        # Split data
        all_labels = list(all_labels)
        random.shuffle(all_labels)
        split_index = int(len(all_labels) * split_ratio)
        train_labels, val_labels = all_labels[:split_index], all_labels[split_index:]

    # Dump data
    def _dump_one_split(split_labels: list, dump_name: str):
        print(f"Dumping {dump_name} data")
        if max_workers >= 1:
            meta_to_dump = _dump_helper_multiview(
                split_labels, split_mapping, root_dict, max_workers=max_workers
            )
        else:
            meta_to_dump = _dump_helper_multiview_single_thread(
                split_labels, split_mapping, root_dict
            )
        path_to_save = os.path.join(openlane_root, f"{dump_name}.pkl")
        if filter_empty_gt:
            print("Filtering empty gt")
            meta_to_dump_filtered = {}
            for identifier, meta_single in tqdm(
                meta_to_dump.items(), desc="Filtering empty gt"
            ):
                if len(meta_single["gt_boxes"]) > 0:
                    meta_to_dump_filtered[identifier] = meta_single
            meta_to_dump = meta_to_dump_filtered
        print(f"Dumping training data to {path_to_save}")
        io.pickle_dump(path_to_save, meta_to_dump)

    if to_dump == "both" or to_dump == "train":
        dump_name = (
            f"training_{split_config['suffix']}"
            if "suffix" in split_config and split_config["suffix"] is not None
            else "training"
        )
        _dump_one_split(train_labels, dump_name)
    if to_dump == "both" or to_dump == "val":
        dump_name = (
            f"validation_{split_config['suffix']}"
            if "suffix" in split_config and split_config["suffix"] is not None
            else "validation"
        )
        _dump_one_split(val_labels, dump_name)
