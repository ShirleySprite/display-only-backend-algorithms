import pandas as pd

from backend_algorithms.export_model import ImageExportExecutor


def trans(input_json):
    exe = ImageExportExecutor(input_json)

    exe.target_path.mkdir(parents=True, exist_ok=True)
    (
        pd.DataFrame(
            [
                (single_data.data_id, single_data.org_path.name, not single_data.get_results().instances)
                for single_data in exe.iter_dataset()
            ],
            columns=["data_id", "data_name", "is_empty"]
        )
        .sort_values(by=["is_empty", "data_id"])
        .to_excel((exe.target_path / "is_empty.xlsx"), index=False)
    )
