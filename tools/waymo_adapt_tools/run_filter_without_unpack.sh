python unified_perception/tools/waymo_adapt_tools/generate_pkl_waymo.py --workers 80 --filter_empty_gt --suffix filtered
python unified_perception/tools/waymo_adapt_tools/create_waymo_gt_bin.py --pkl-path data/waymo/openlane_format/validation_filtered.pkl --bin-name cam_gt_filtered.bin