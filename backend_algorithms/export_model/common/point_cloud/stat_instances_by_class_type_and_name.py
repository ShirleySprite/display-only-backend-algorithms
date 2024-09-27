import pandas as pd

from backend_algorithms.export_model import Lidar3DExportExecutor


def trans(input_json):
    exe = Lidar3DExportExecutor(input_json)

    stat = []
    for x in exe.iter_dataset():
        r = x.get_results()
        cnt_df = pd.DataFrame(
            [(x.type, x.class_name) for x in r.instances + r.segments],
            columns=["class_type", "class_name"]
        ).groupby(by=["class_type", "class_name"]).agg({"class_name": "count"}).rename(
            columns={"class_name": "count"}
        )
        cnt_df["data_id"] = x.data_id
        cnt_df.reset_index(inplace=True)
        stat.append(cnt_df)

    stat_df = pd.concat(stat).sort_values(by=["data_id", "class_name", "class_type"]).reset_index(drop=True)
    pivot_stat_df = pd.pivot(stat_df.set_index(["data_id", "class_type"]), columns="class_name",
                             values="count").fillna(0).astype(int)
    total_df = pd.pivot_table(stat_df, values="count", index="class_type", columns="class_name", aggfunc="sum")

    exe.target_path.mkdir(parents=True, exist_ok=True)
    pivot_stat_df.to_excel(exe.target_path / "single_data_stat.xlsx")
    total_df.to_excel(exe.target_path / "total_data_stat.xlsx")
