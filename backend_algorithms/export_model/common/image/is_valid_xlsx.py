import pandas as pd

from backend_algorithms.export_model import ImageExportExecutor


def trans(input_json):
    exe = ImageExportExecutor(input_json)

    exe.target_path.mkdir(parents=True, exist_ok=True)
    (
        pd.DataFrame(
            [
                (
                    single_data.data_id,
                    single_data.org_path.name,
                    '"validity":"INVALID",' not in single_data.result_path.read_text(encoding="utf-8")
                )
                for single_data in exe.iter_dataset()
            ],
            columns=["data_id", "data_name", "is_valid"]
        )
        .sort_values(by=["is_valid", "data_id"])
        .to_excel((exe.target_path / "is_valid.xlsx"), index=False)
    )
