echo "trying pre-compiled waymo metrics"
unified_perception/tools/waymo_utils/compute_detection_metrics_main
echo "converting waymo dataset"
python unified_perception/tools/waymo_adapt_tools/unpack_waymo.py --workers 80
python unified_perception/tools/waymo_adapt_tools/generate_pkl_waymo.py --workers 80
python unified_perception/tools/waymo_adapt_tools/create_waymo_gt_bin.py

