import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from backend_algorithms.import_model.import_utils.import_dataset import ImportLidarDataset
from backend_algorithms.utils.lidar import transform_matrix, header


def trans(input_json):
    origin_path = Path(input_json.get('originPath'))
    target_path = Path(input_json.get('targetPath'))

    direc_map = {
        "front_camera": 0,
        "left_camera": 1,
        "back_camera": 2,
        "right_camera": 3,
        "front_left_camera": 4,
        "front_right_camera": 5
    }
    n_direcs = len(direc_map)

    for pkl_data in ImportLidarDataset(origin_path, "**/lidar/*.pkl"):
        pkl_file = pkl_data.meta_path
        pc_df = pickle.loads(pkl_file.read_bytes())
        xyzi_df = pc_df[["x", "y", "z", "i"]].astype(np.float32)
        xyzi_df["i"] = xyzi_df["i"].astype(int)

        n_points = len(xyzi_df)
        header_data = header.format(n_points, n_points)
        pcd_data = header_data.encode() + xyzi_df.to_records(index=False).tobytes()

        pcd_output = pkl_data.prepare_target_path(target_path).with_suffix(".pcd")
        pcd_output.write_bytes(pcd_data)

        for cur_direc, cur_direc_i in direc_map.items():
            img_path = pkl_file.parent.parent / "camera" / cur_direc / pkl_file.with_suffix(".jpg").name

            if not img_path.exists():
                continue

            img_output = pcd_output.parent.parent / f"camera_image_{cur_direc_i}" / pcd_output.with_suffix(
                ".jpg").name
            img_output.parent.mkdir(parents=True, exist_ok=True)
            img_output.write_bytes(img_path.read_bytes())

        x1_config = [{}] * n_direcs
        for cur_direc, cur_direc_i in direc_map.items():
            int_path = pkl_file.parent.parent / "camera" / cur_direc / "intrinsics.json"
            ext_path = pkl_file.parent.parent / "camera" / cur_direc / "poses.json"

            int_config = json.loads(int_path.read_text(encoding="utf-8"))
            ext_config = json.loads(ext_path.read_text(encoding="utf-8"))[int(pkl_file.stem)]

            t = list(ext_config["position"].values())
            quat = ext_config["heading"]
            r = R.from_quat([quat['x'], quat['y'], quat['z'], quat['w']]).as_matrix()
            trans_mat = np.linalg.inv(transform_matrix(t, r))

            new_config = {
                "camera_internal": int_config,
                "camera_external": trans_mat.ravel().tolist(),
            }
            x1_config[cur_direc_i] = new_config

        config_output = pcd_output.parent.parent / "camera_config" / pcd_output.with_suffix(".json").name
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_text(json.dumps(x1_config), encoding="utf-8")
