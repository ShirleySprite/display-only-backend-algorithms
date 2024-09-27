import pandas as pd

from backend_algorithms.export_model import AVExportExecutor


def trans(
        input_json
):
    stat = {}
    exe = AVExportExecutor(input_json)
    for x in exe.iter_dataset():
        for y in x.get_results().instances:
            annotator = y.created_by
            stat.setdefault(annotator, 0)
            stat[annotator] += y.length

    exe.target_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stat.items(), columns=["annotator", "length"]).to_excel(exe.target_path / "length.xlsx", index=False)
