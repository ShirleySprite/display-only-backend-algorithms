from typing import List, Dict, Union, Optional, Tuple

import numpy as np


def basic_bbox_to_yolo(
        bbox,
        iw,
        ih
):
    total_x = [x['x'] for x in bbox if x]
    total_y = [x['y'] for x in bbox if x]
    xmin, ymin = min(total_x), min(total_y)
    xmax, ymax = max(total_x), max(total_y)
    xc = ((xmax + xmin) / 2) / iw
    yc = ((ymax + ymin) / 2) / ih
    w = (xmax - xmin) / iw
    h = (ymax - ymin) / ih

    return [xc, yc, w, h]


def norm_coords(
        points: List,
        iw: int,
        ih: int
) -> List:
    if isinstance(points, Dict):
        result = [
            v / div
            for v, div in zip([points['x'], points['y']], [iw, ih])
        ]

    elif isinstance(points[0], (np.number, int, float)):
        result = [points[0] / iw, points[1] / ih]

    else:
        result = [norm_coords(x, iw, ih) for x in points]

    return result


def find_diagonal(
        points_list: List,
        return_w_h: bool = False,
        keys: Optional[List] = None
) -> Union[List, Dict, None]:
    if isinstance(points_list[0], Dict):
        x_list = [p['x'] for p in points_list if p]
        y_list = [p['y'] for p in points_list if p]
    elif isinstance(points_list[0], (List, Tuple)):
        x_list = [p[0] for p in points_list if p]
        y_list = [p[1] for p in points_list if p]
    else:
        return None

    xmin = min(x_list)
    ymin = min(y_list)
    xmax = max(x_list)
    ymax = max(y_list)

    result = (xmin, ymin, xmax, ymax) if not return_w_h else (xmin, ymin, xmax - xmin, ymax - ymin)
    result = {keys[i]: v for i, v in enumerate(result)} if keys else result

    return result


def gen_basic_points(
        xmin: Union[int, float],
        ymin: Union[int, float],
        xmax: Union[int, float],
        ymax: Union[int, float]
):
    return [
        {
            'x': xmin,
            'y': ymin
        },
        {
            'x': xmax,
            'y': ymin
        },
        {
            'x': xmax,
            'y': ymax
        },
        {
            'x': xmin,
            'y': ymax
        },
    ]


def points_to_list(
        pts: Union[List, Dict, int, float]
):
    result = []

    if isinstance(pts, Dict):
        pts = [v for k, v in pts.items() if k in ['x', 'y', 'z']]

    if isinstance(pts, List):
        for x in pts:
            new_x = points_to_list(x)
            if new_x != []:
                result.append(new_x)
        return result

    return pts


def round_points(
        pts: Union[List, Dict],
        round_param: Optional[int] = None
):
    if isinstance(pts, List):
        return [round_points(x, round_param) for x in pts]

    if isinstance(pts, Dict):
        return {k: round_points(v, round_param) for k, v in pts.items()}

    if isinstance(pts, (int, float)):
        return round(pts, round_param)


def is_clockwise(
        pts: List[Union[List, Dict]],
        y_down: bool = True
):
    if isinstance(pts[0], Dict):
        pts = [list(pt.values()) for pt in pts]

    n = len(pts)
    s = 0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % n]
        s += 0.5 * (x1 * y2 - x2 * y1)

    if y_down:
        s = -s

    return s < 0


def id_to_rgb(
        id_map: Union[int, np.ndarray]
) -> Union[List, np.ndarray]:
    """
    Converts unique ID to RGB color.
    """
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map

    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def angle_cos(
        points: List[List[Union[int, float]]]
):
    n = len(points)
    vecs = []
    for i in range(n):
        pt1 = points[i]
        pt2 = points[(i + 1) % n]

        vecs.append(
            [pt2[0] - pt1[0], pt2[1] - pt1[1]]
        )

    result = []
    for i in range(n):
        vec1 = np.array(vecs[i])
        vec2 = np.array(vecs[(i + 1) % n])

        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        result.append(cos_theta)

    return result


def sort_coco(
        coco_result: Dict
) -> None:
    coco_result["images"].sort(key=lambda x: x["id"])
    coco_result["annotations"].sort(key=lambda x: x["image_id"])


def gen_coco_ann_id(
        coco_result: Dict
) -> None:
    ann_id = 1
    for x in coco_result["annotations"]:
        x["id"] = ann_id
        ann_id += 1
