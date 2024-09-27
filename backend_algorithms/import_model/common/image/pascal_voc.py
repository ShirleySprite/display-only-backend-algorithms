import json
from pathlib import Path
from typing import List, Dict, Union, Callable, Optional, Generator, Iterable, Tuple
from collections import defaultdict

import xmltodict


def batched(
        iterable: Iterable,
        n: int
) -> Generator[Tuple, None, None]:
    from itertools import islice

    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def gen_basic_points(xmin, ymin, xmax, ymax):
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


def trans(input_json):
    code = 'OK'
    message = ''

    origin_path = Path(input_json.get('originPath'))
    target_path = Path(input_json.get('targetPath'))
    clss = input_json.get('datasetClassList', [])

    class_map = {x['name']: x['id'] for x in clss}
    attr_map = defaultdict(dict)
    for x in clss:
        attrs = x['attributes']
        for attr in attrs:
            attr_map[x['name']][attr['name']] = attr['id']

    # 找到图片
    for img_file in origin_path.rglob('**/ImageSets/**/*.*'):
        if not img_file.is_file():
            continue

        # 复制图片
        img_output = target_path / img_file.relative_to(origin_path)
        img_output.parent.mkdir(parents=True, exist_ok=True)
        img_file.replace(img_output)

        # 对应xml
        xml_file = Path(str(img_file).replace('ImageSets', 'Annotations')).with_suffix('.xml')
        if not xml_file.exists():
            continue

        # 生成结果
        org_result = xmltodict.parse(xml_file.open('rb'))
        objects = []
        xml_objs = org_result['annotation'].get('object', [])
        if isinstance(xml_objs, dict):
            xml_objs = [xml_objs]
        for obj in xml_objs:
            class_name = obj['name']
            new_obj = {
                "className": class_name,
                "classId": class_map.get(class_name),
                "classValues": []
            }

            # 加入属性，只支持1层的属性
            attr = attr_map.get(class_name)
            if attr:
                for k, v in attr.items():
                    new_obj['classValues'].append(
                        {
                            "id": v,
                            "isLeaf": True,
                            "name": k,
                            "value": obj.get(k)
                        }
                    )

            # 矩形框
            if 'bndbox' in obj:
                new_obj['type'] = 'BOUNDING_BOX'
                bndbox = obj['bndbox']
                xmin, ymin, xmax, ymax = bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']
                new_obj['contour'] = {'points': gen_basic_points(xmin, ymin, xmax, ymax)}

            # 多边形
            elif 'polygon' in obj:
                new_obj['type'] = 'POLYGON'
                poly = obj['polygon']
                new_obj['contour'] = {
                    'points': [{'x': float(vx), 'y': float(vy)} for (_, vx), (_, vy) in batched(poly.items(), 2)]}

            # 折线
            elif 'line' in obj:
                new_obj['type'] = 'POLYLINE'
                line = obj['line']
                new_obj['contour'] = {
                    'points': [{'x': float(vx), 'y': float(vy)} for (_, vx), (_, vy) in batched(line.items(), 2)]}

            # 点
            elif 'point' in obj:
                new_obj['type'] = 'KEY_POINT'
                p = obj['point']
                x, y = float(p['x']), float(p['y'])
                new_obj['contour'] = {'points': [{'x': x, 'y': y}]}

            # 圆
            elif 'circle' in obj:
                new_obj['type'] = 'CIRCLE'
                circle = obj['circle']
                cx, cy, r = float(circle['cx']), float(circle['cy']), float(circle['r'])
                new_obj['contour'] = {'points': [{'x': cx, 'y': cy}], 'radius': r}

            else:
                continue

            objects.append(new_obj)

        # 保存结果
        json_output = img_output.with_suffix('.json')
        json_output = json_output.parent / 'result' / json_output.name
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json.dump({'instances': objects}, json_output.open('w', encoding='utf-8'), ensure_ascii=False)

    return code, message
