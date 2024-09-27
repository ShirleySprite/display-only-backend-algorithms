import time
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from loguru import logger

from backend_algorithms.utils.image import find_diagonal
from backend_algorithms.utils.general import groupby


def to_array_points(
        basic_points
):
    return np.array([list(x.values()) for x in basic_points if x != {}])


class OntoParser:
    def __init__(
            self,
            onto_clss
    ):
        skeletons, others = [], []
        for x in onto_clss:
            if x['toolType'] == 'SKELETON':
                skeletons.append(x)
            elif x['toolType'] in ['BOUNDING_BOX', 'POLYGON']:
                others.append(x)

        self._skeletons = self._parse_skeleton(skeletons)
        self.onto = self._parse_others(others)
        self._merge_onto()

    @staticmethod
    def _parse_skeleton(
            skeletons
    ):
        result = {}
        for x in skeletons:
            class_name = x['name']
            result[class_name] = {
                'supercategory': class_name,
                'id': 0,
                'name': class_name,
                'keypoints': [p['label'] for p in x['toolTypeOptions']['skeletonConfig']['pointList']],
                'skeleton': [list(line.values()) for line in x['toolTypeOptions']['skeletonConfig']['lineList']]
            }

        return result

    @staticmethod
    def _parse_others(
            others
    ):
        result = {}
        for x in others:
            class_name = x['name']
            result[class_name] = {
                'supercategory': class_name,
                'id': 0,
                'name': class_name,
            }

        return result

    def _merge_onto(
            self
    ):
        self.onto.update(self._skeletons)
        for i, k in enumerate(self.onto.keys()):
            self.onto[k]['id'] = i + 1


def trans(input_json):
    code = 'OK'
    message = ''

    origin_path = Path(input_json.get('originPath'))
    target_path = Path(input_json.get('targetPath'))
    has_origin_file = input_json.get('hasOriginFile')

    # 起始时间
    start_time = time.time()

    # 解析ontology
    clss = input_json.get('datasetClassList')
    try:
        op = OntoParser(clss)
    except:
        code = 'ERROR'
        message = "ontology doesn't match coco format"

        return code, message

    # 读取所有result json, 按result的上一级分开存放
    total_files = origin_path.rglob('*.json')
    files_map = groupby(total_files, lambda x: x.parent.name)
    result_map = {}
    for k, v in files_map.items():
        result_map[k] = groupby(v, lambda x: x.parent.parent)

    # 记录图片编号
    img_id = 1
    ann_id = 1

    logger.info(f'time(s): {time.time() - start_time:.3f}; '
                f'script: export_model/common/image/coco; '
                f'info: preparation finished')

    # 导出
    for folder, files in result_map['result'].items():
        # 当前结果框架
        cur_result = {
            'info': {},
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': list(op.onto.values())
        }

        # images字段
        datas = result_map['data'][folder]
        # 记录当前json里的图片id
        img_id_map = {}
        for d in datas:
            meta_data = json.load(d.open(encoding='utf-8'))['images'][0]
            file_name = meta_data['filename']

            cur_img_data = {
                "file_name": file_name,
                "height": meta_data['height'],
                "width": meta_data['width'],
                "id": img_id
            }
            cur_result['images'].append(cur_img_data)

            # 记录图片编号
            img_id_map[d.stem] = img_id
            img_id += 1

            if has_origin_file:
                img_path = d.parent.parent / file_name
                if not img_path.exists():
                    img_path = d.parent.parent / 'image_0' / file_name

                img_output = target_path / folder.relative_to(origin_path) / 'images' / img_path.name
                img_output.parent.mkdir(parents=True, exist_ok=True)
                img_path.replace(img_output)

            logger.info(f'time(s): {time.time() - start_time:.3f}; '
                        f'script: export_model/common/image/coco; '
                        f'info: parse data {d} finished')

        # annotations字段
        for file in files:
            # 对应回原数据
            img_id = img_id_map[file.stem]

            ann_dict = json.load(file.open(encoding='utf-8'))

            # 整合多个resource的标注
            total_anns = []
            for single_source in ann_dict:
                total_anns.extend(single_source['instances'])

            # 改单个格式
            # 区分框/多边形、骨骼点
            box_instances = [x for x in total_anns if x['type'] in ["BOUNDING_BOX", "RECTANGLE"]]
            others_instances = [x for x in total_anns if x['type'] in ['POLYGON', 'SKELETON']]

            # 框单独导
            for box_inst in box_instances:
                # 获取类别
                cur_class_name = box_inst['className'] or box_inst['modelClass']
                xmin, ymin, w, h = find_diagonal(
                    box_inst['contour']['points'],
                    return_w_h=True
                )

                cur_coco_ann = {
                    "area": box_inst['contour'].get('area') or (round(w * h)),
                    "image_id": img_id,
                    "bbox": [xmin, ymin, w, h],
                    "iscrowd": 0,
                    "segmentation": [],
                    "category_id": op.onto.get(cur_class_name, {'id': -1})['id'],
                    "id": ann_id
                }

                cur_result['annotations'].append(cur_coco_ann)
                ann_id += 1

            # 将同组结果聚合, 没编组的自己一组
            group_dict = groupby(others_instances, lambda x: x['groups'][0] if x['groups'] else x['id'])

            for gid, instances in group_dict.items():
                # 获取类别
                cur_class_name = instances[0]['className'] or instances[0]['modelClass']

                # 区分polygon和skeleton
                polygons = [to_array_points(x['contour']['points']) for x in instances if x['type'] == 'POLYGON']
                skeleton = ([x for x in instances if x['type'] == 'SKELETON'] + [{}])[0]

                # 创建MultiPolygon
                mp = MultiPolygon([Polygon(x) for x in polygons])
                xmin, ymin, xmax, ymax = mp.bounds

                cur_coco_ann = {
                    "segmentation": [x.flatten().tolist() for x in polygons],
                    "area": mp.area,
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "category_id": op.onto.get(cur_class_name, {'id': -1})['id'],
                    "id": ann_id
                }

                # 加Keypoints字段进去
                if skeleton:
                    sk = skeleton
                    cur_coco_ann['num_keypoints'] = len([x for x in sk['contour']['nodes'] if x['attr']['valid']])
                    cur_coco_ann['keypoints'] = []
                    for n in skeleton['contour']['nodes']:
                        try:
                            cur_nodes = [n['position']['x'], n['position']['y'], int(n['attr']['code'])]
                        except ValueError:
                            raise ValueError('skeleton node attribute are numbers like 1 or "1"')

                        cur_coco_ann['keypoints'] += cur_nodes

                cur_result['annotations'].append(cur_coco_ann)
                ann_id += 1

            logger.info(f'time(s): {time.time() - start_time:.3f}; '
                        f'script: export_model/common/image/coco; '
                        f'info: parse annotation {file} finished')

        coco_output = target_path / folder.relative_to(origin_path) / (folder.name + '.json')
        coco_output.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cur_result, coco_output.open('w', encoding='utf-8'), ensure_ascii=False)

    return code, message
