import pandas as pd

from backend_algorithms.export_model import Lidar3DExportExecutor


def trans(input_json):
    exe = Lidar3DExportExecutor(input_json)

    stat = []
    for x in exe.iter_dataset():
        cur_total = 0
        for y in x.get_results().instances:
            if y.type == "3D_LANE_POLYLINE":
                cur_total += y.length

        stat.append((x.data_id, cur_total))

    stat_df = pd.DataFrame(stat, columns=["data_id", "length"]).set_index("data_id").sort_index()

    exe.target_path.mkdir(parents=True, exist_ok=True)
    stat_df.to_excel(exe.target_path / "total_length.xlsx")
