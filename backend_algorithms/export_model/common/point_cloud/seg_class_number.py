import pandas as pd

from backend_algorithms.export_model import Lidar3DExportExecutor
from backend_algorithms.utils.lidar import PointCloud


def trans(input_json):
    exe = Lidar3DExportExecutor(input_json)
    exe.copy_export_files()
    id_number_map = exe.ontology_info.id_number_map

    for single_data in exe.iter_dataset():
        no_number_map = {x.no: id_number_map[x.class_id] for x in single_data.get_results().segments}
        seg_df = pd.DataFrame(single_data.get_results().segmentation.data).applymap(lambda no: no_number_map.get(no, 0))
        PointCloud.save_pcd(
            seg_df.to_records(index=False), exe.target_path / single_data.segmentation_file.relative_to(exe.origin_path)
        )
