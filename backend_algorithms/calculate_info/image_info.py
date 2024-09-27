from PIL import Image

import numpy as np
from shapely.geometry import Polygon

from backend_algorithms.utils.general import groupby


def cal_instances_info(
        instances
):
    results = []
    for obj in instances:
        if obj.get('type') not in ['RECTANGLE', 'BOUNDING_BOX', 'POLYGON', 'POLYGON_PLUS']:
            continue

        contour = obj.get('contour') or {}
        pts = [
            [x.get('x', 0), x.get('y', 0)]
            for x in contour.get('points') or []
            if x
        ]
        holes = [
            [[x.get('x', 0), x.get('y', 0)] for x in (hole.get('points') or []) if x]
            for hole in contour.get('interior') or []
        ]

        try:
            poly = Polygon(
                shell=pts,
                holes=holes
            )
            area = round(poly.area)
        except ValueError:
            area = 0

        obj_info = {
            "objectId": obj['id'],
            "area": area
        }
        results.append(obj_info)

    return results


def cal_segments_info(
        segments
):
    results = []
    for mask_seg_file, cur_segments in groupby(
            segments,
            func=lambda x: x["segmentResultFilePath"]
    ).items():
        img_array = np.array(Image.open(mask_seg_file), dtype=np.uint32)
        mask_array = img_array[:, :, 0] + img_array[:, :, 1] * 256 + img_array[:, :, 2] * 256 * 256

        for seg in segments:
            obj_info = {
                "objectId": seg['id'],
                "area": int(np.sum(mask_array == seg['no']))
            }
            results.append(obj_info)

    return results


def cal_image_info(
        data
):
    return cal_instances_info(data["instances"]) + cal_segments_info(data["segments"])
