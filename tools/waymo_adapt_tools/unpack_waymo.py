# unpack waymo data and arrange them in a way that is similar to openlanev1

import os
from os import path as osp
import argparse

# arg parser
parser = argparse.ArgumentParser(description="Generate pickle file")
parser.add_argument(
    "--load_dir", type=str, default="data/waymo/waymo_format", help="load dir"
)
parser.add_argument(
    "--save_dir", type=str, default="data/waymo/openlane_format", help="save dir"
)
parser.add_argument("--split", type=str, default="both", help="collection name")
parser.add_argument("--workers", type=int, default=4, help="multiprocessing workers")
parser.add_argument("--verbose", type=bool, default=False, help="verbose mode")


# add the path of the project to sys.path
import sys

sys.path.insert(0, osp.join(os.getcwd(), "./data_converter"))
from data_converter import waymo_converter as waymo

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Waymo unpacking for cooperating with openlanev1, setting: {args}")
    print("Note: We recommend waymo 1.4.2 for openlanev1 cooperation.")
    print("Converting...")
    print(args)

    # generate pickle file
    load_dir = args.load_dir  # e.g. osp.join(root_path, 'waymo_format')
    save_dir = args.save_dir  # e.g. osp.join(root_path, 'openlane_format')
    splits = ["training", "validation"]
    # only image and lidar data are converted, if you need to do detection, you need to consider what kind of calib, pose, label, etc. you need especially the format.
    for split in splits:
        print(f"Converting {split} set...")
        converter = waymo.Waymo2Waymo(
            load_dir,
            save_dir,
            split,
            prefix="",
            workers=args.workers,
            test_mode=False,
            verbose=args.verbose,
        )
        converter.convert()

    print("Done.")
