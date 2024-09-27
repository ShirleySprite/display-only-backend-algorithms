import numpy as np
import cv2  # noqa
from PIL import Image

from backend_algorithms.export_model import ImageExportExecutor
from backend_algorithms.utils.general import hex_to_rgb


def draw_single_instance(
        img,
        inst,
        color_dict
):
    t = inst.type
    if t in ['BOUNDING_BOX', 'POLYGON', 'POLYLINE', 'KEY_POINT']:
        color = hex_to_rgb(color_dict.get(inst.class_id, '#FF0000'))

        if t == "BOUNDING_BOX":
            draw_func = cv2.polylines  # noqa
            params = {
                "img": img,
                "pts": [np.array(inst.points_to_list()).round().astype(int)],
                "isClosed": True,
                "color": color,
                "thickness": 2
            }

        elif t == "POLYGON":
            draw_func = cv2.polylines  # noqa
            outer, inner = inst.points_to_list()
            outer = np.array(outer).round().astype(int)
            inner = [np.array(inn).round().astype(int) for inn in inner]
            params = {
                "img": img,
                "pts": [outer, *inner],
                "isClosed": True,
                "color": color,
                "thickness": 2
            }

        elif t == "POLYLINE":
            draw_func = cv2.polylines  # noqa
            params = {
                "img": img,
                "pts": [np.array(inst.points_to_list()).round().astype(int)],
                "isClosed": False,
                "color": color,
                "thickness": 2
            }

        else:
            draw_func = cv2.circle  # noqa
            params = {
                "img": img,
                "center": [round(inst.x), round(inst.y)],
                "radius": 2,
                "color": color,
                "thickness": -1
            }

        draw_func(**params)


def _convert(
        single_data,
        color_dict
):
    img = np.asarray(Image.open(single_data.file).convert('RGB')).copy()

    for y in single_data.get_results().instances:
        draw_single_instance(
            img=img,
            inst=y,
            color_dict=color_dict
        )

    return Image.fromarray(img)


def trans(input_json):
    exe = ImageExportExecutor(input_json)
    color_dict = exe.ontology_info.id_color_map
    exe.convert_all_data(_convert, result_ext=".png", color_dict=color_dict)
