import numpy as np
import pandas as pd

from backend_algorithms.export_model import Lidar3DExportExecutor


def trans(input_json):
    exe = Lidar3DExportExecutor(input_json)
    exe.copy_export_files()

    for single_data in exe.iter_dataset():
        pc = single_data.get_org_pcd()
        pc_df = pd.DataFrame(pc.data)
        if single_data.segmentation_file:
            pc_df["seg"] = pd.DataFrame(single_data.get_results().segmentation.data)["seg"]
        else:
            pc_df["seg"] = 0
        pc_df["seg"] = pc_df["seg"].astype(np.int32)
        pc.save_pcd(
            pc_df.to_records(index=False),
            exe.target_path / single_data.segmentation_file.relative_to(exe.origin_path)
        )
