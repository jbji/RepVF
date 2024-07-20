import numpy as np
import os

pseudo_idx_counter = {}


def npdump(np_obj, img_metas, out_type="default_objects"):
    global pseudo_idx_counter

    if os.environ.get("SAVE_FOR_VISUALIZATION") == "True":
        if out_type not in pseudo_idx_counter:
            pseudo_idx_counter[out_type] = 0

        dump_limit = int(os.environ.get("DUMP_LIMIT", 1000))  # Default limit
        dump_step = int(os.environ.get("DUMP_STEP", 5))  # Default step

        output_dir = os.environ.get(
            "OUTPUT_DIR", "./visualizations/" + out_type
        )  # Default to current directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = img_metas[0]["sample_idx"] + ".npy"  # Default filename
        output_path = os.path.join(output_dir, file_name)

        pseudo_idx = pseudo_idx_counter[out_type]

        if pseudo_idx % dump_step == 0 and pseudo_idx < dump_limit:
            np.save(output_path, np_obj)

        # Increment the pseudo index for this out_type
        pseudo_idx_counter[out_type] += 1

    return None


def npdump_nocheck(np_obj, img_metas, out_type="default_objects"):
    output_dir = os.environ.get(
        "OUTPUT_DIR", "./visualizations/" + out_type
    )  # Default to current directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = img_metas[0]["sample_idx"] + ".npy"  # Default filename
    output_path = os.path.join(output_dir, file_name)

    np.save(output_path, np_obj)

    return None
