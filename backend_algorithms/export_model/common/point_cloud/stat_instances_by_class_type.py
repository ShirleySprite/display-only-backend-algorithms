from collections import Counter

import pandas as pd

from backend_algorithms.export_model import Lidar3DExportExecutor


def trans(input_json):
    exe = Lidar3DExportExecutor(input_json)

    stat = []
    data_ids = []
    for x in exe.iter_dataset():
        r = x.get_results()
        stat.append(Counter([x.type for x in r.instances + r.segments]))
        data_ids.append(x.data_id)

    stat_df = pd.DataFrame(stat, index=data_ids).fillna(0).astype(int).sort_index()
    stat_df.loc['Total'] = stat_df.sum()

    exe.target_path.mkdir(parents=True, exist_ok=True)
    stat_df.to_excel(exe.target_path / "stat_instances_by_class_type.xlsx")
