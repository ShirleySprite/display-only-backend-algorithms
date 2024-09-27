import json
from pathlib import Path
from functools import reduce
from typing import List, Union, Optional

from PIL import Image
import numpy as np


def filter_parts(
        org_path: Union[str, Path],
        filter_list: Optional[List] = None
) -> Path:
    if filter_list is None:
        filter_list = ['image_0']

    if isinstance(org_path, str):
        org_path = Path(org_path)

    return Path(*[x for x in org_path.parts if x not in filter_list])


def hex_to_rgb(
        hex_code
):
    # 去除十六进制代码中的 '#' 符号（如果有）
    hex_code = hex_code.strip('#')

    # 将十六进制代码分割成 R、G、B 三部分
    red = int(hex_code[0:2], 16)
    green = int(hex_code[2:4], 16)
    blue = int(hex_code[4:6], 16)

    return red, green, blue


def fill_box(
        box,
        mask_data,
        color
):
    mask_data = np.array(mask_data)
    for start_id, cur_len in mask_data.reshape(-1, 2):
        box[start_id: start_id + cur_len] = color


def trans(input_json):
    origin_path = Path(input_json.get('originPath'))
    target_path = Path(input_json.get('targetPath'))

    clss = input_json.get('datasetClassList')
    color_map = {c['id']: hex_to_rgb(c['color']) for c in clss}

    for meta_file in origin_path.rglob("**/data/*.json"):
        # meta data
        meta_data = json.load(meta_file.open(encoding='utf-8'))
        img_info = meta_data['images'][0]
        org_path = filter_parts(img_info['zipPath'] or img_info['filename'])
        iw, ih = img_info['width'], img_info['height']

        # result
        result_file = meta_file.parent.parent / 'result' / meta_file.name
        if result_file.exists():
            result_data = json.load(result_file.open(encoding='utf-8'))
        else:
            result_data = [{'segments': []}]

        # 合并所有source结果
        total_ann = reduce(lambda x, y: x + y['segments'], result_data, [])

        # 背景板
        result_img = np.zeros((ih, iw, 3), dtype=np.uint8)

        for ann in total_ann:
            # 加入单个标注
            xmin, ymin, width, height = ann['contour']['box']
            box_1d = np.zeros((width * height, 3), dtype=np.uint8)
            color = color_map[ann['classId']]
            fill_box(
                box=box_1d,
                mask_data=ann['contour']['maskData'],
                color=color
            )
            box_2d = box_1d.reshape(height, width, 3)

            result_img[ymin: ymin + height, xmin: xmin + width] += box_2d

        # 保存
        img_output = target_path / org_path.with_suffix('.png')
        img_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_img).save(img_output)
