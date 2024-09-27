from backend_algorithms.export_model import ImageExportExecutor


def trans(input_json):
    exe = ImageExportExecutor(input_json)
    exe.del_invalid_data()
    for x in exe.iter_dataset():
        x_output = exe.target_path / x.org_path
        x_output.parent.mkdir(parents=True, exist_ok=True)
        x.file.replace(x_output)
