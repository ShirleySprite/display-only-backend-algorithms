import numpy as np
from PIL import Image

from backend_algorithms.export_model import ImageExportExecutor


def _convert(
        single_data,
        color_map
):
    return Image.fromarray(
        single_data.get_results().gen_mask(
            img=np.zeros(shape=(single_data.ih, single_data.iw), dtype=np.uint8),
            color_map=color_map
        )
    )


def trans(input_json):
    exe = ImageExportExecutor(input_json)
    color_map = exe.ontology_info.id_number_map
    exe.convert_all_data(_convert, result_ext=".png", color_map=color_map)
