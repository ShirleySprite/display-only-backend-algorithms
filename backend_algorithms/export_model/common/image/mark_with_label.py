import numpy as np
import cv2  # noqa
from PIL import Image

from backend_algorithms.export_model import ImageExportExecutor
from backend_algorithms.utils.general import hex_to_rgb
from backend_algorithms.export_model.common.image.mark_without_label import draw_single_instance


def cal_proper_coord(
        org_x,
        org_y,
        iw,
        ih,
        x_interval=10,
        y_interval=10
):
    new_x = min(max(0, org_x), iw - x_interval)
    new_y = min(max(0, org_y), ih - y_interval)

    return new_x, new_y


def _convert(
        single_data,
        color_dict
):
    img = np.asarray(Image.open(single_data.file).convert('RGB')).copy()
    iw, ih = single_data.iw, single_data.ih

    for y in single_data.get_results().instances:
        if y.type in ['BOUNDING_BOX', 'POLYGON', 'POLYLINE', 'KEY_POINT']:
            draw_single_instance(
                img=img,
                inst=y,
                color_dict=color_dict
            )

            color = hex_to_rgb(color_dict.get(y.class_id, '#FF0000'))
            cv2.putText(  # noqa
                img=img,
                text=f"{y.track_name}-{y.class_name}",
                org=cal_proper_coord(
                    org_x=round(y.points[0]['x']),
                    org_y=round(y.points[0]['y']),
                    iw=iw,
                    ih=ih
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # noqa
                fontScale=1,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA  # noqa
            )

    return Image.fromarray(img)


def trans(input_json):
    exe = ImageExportExecutor(input_json)
    color_dict = exe.ontology_info.id_color_map
    exe.convert_all_data(_convert, result_ext=".png", color_dict=color_dict)
