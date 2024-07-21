# Copyright [2024] Chunliang Li
# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements; and copyright (c) [2024] Chunliang Li;
# This program and the accompanying materials are made available under the
# terms of the Apache License 2.0 which is available at
# http://www.apache.org/licenses/LICENSE-2.0.

# =========================================================
# author: jbji
# date: 2023-03-26
# =========================================================


import argparse

from openlanev1.preprocessing import collect_multiview

# # arg parser
parser = argparse.ArgumentParser(description="Generate pickle file")
# parser.add_argument(
#     "--data_root", type=str, default="data/openlanev1/images", help="data root path"
# )
# parser.add_argument(
#     "--meta_root", type=str, default="data/openlanev1/lane3d_300", help="meta root path"
# )
# parser.add_argument(
#     "--collection", type=str, default="training", help="collection name"
# )

# # point_interval
# parser.add_argument(
#     "--point_interval",
#     type=int,
#     default=1,
#     help="interval for subsampling points of lane lines",
# )

# add workers argument
parser.add_argument(
    "--workers",
    type=int,
    default=16,
    help="number of workers for multiprocessing",
)
parser.add_argument(
    "--filter_empty_gt",
    action="store_true",
    help="whether to filter out frames with empty gt",
)
parser.add_argument(
    "--suffix",
    type=str,
    default=None,
    help="suffix of the pickle file, e.g. 'filtered'",
)
parser.add_argument(
    "--lane3d",
    type=str,
    default="lane3d_1000",
    help="name of the lane3d folder",
)

if __name__ == "__main__":
    args = parser.parse_args()
    # print(f"OpenLaneV1 Preprocessing, setting: {args}")
    # print("Generating pickle file...")
    # print(args)

    # generate pickle file
    collect_multiview(
        root_dict={
            "openlane_format": "data/waymo/openlane_format",
            "detection3d": "detection3d_1000",
            "lane3d": args.lane3d,
        },
        split_config={"suffix": args.suffix, "method": "lane3d"},
        to_dump="both",
        max_workers=args.workers,
        filter_empty_gt=args.filter_empty_gt,
    )

    print("Done.")
