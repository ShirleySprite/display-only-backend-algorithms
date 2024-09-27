import numpy as np
from PIL import Image

from backend_algorithms.export_model import ImageExportExecutor
from backend_algorithms.utils.general import hex_to_rgb


def _convert(
        single_data,
        color_map
):
    return Image.fromarray(
        single_data.get_results().gen_mask(
            img=np.zeros(shape=(single_data.ih, single_data.iw, 3), dtype=np.uint8),
            color_map=color_map
        )
    )


def trans(input_json):
    exe = ImageExportExecutor(input_json)
    color_map = {k: hex_to_rgb(v) for k, v in exe.ontology_info.id_color_map.items()}
    exe.convert_all_data(_convert, result_ext=".png", color_map=color_map)
