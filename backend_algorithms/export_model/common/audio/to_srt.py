from backend_algorithms.export_model import AVExportExecutor


def _convert(single_data):
    r = ''
    for i, y in enumerate(single_data.get_results().instances):
        start, end = y.to_hms()
        r += (f'{i + 1}\n{start} --> '
              f'{end}\n{y.class_name}\n\n')

    return r


def trans(input_json):
    AVExportExecutor(input_json).convert_all_data(_convert, result_ext=".srt")
