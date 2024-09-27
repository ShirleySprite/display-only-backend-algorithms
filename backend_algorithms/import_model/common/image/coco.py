#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import time
from pathlib import Path
from shutil import move
from typing import List, Dict, Union, Optional
from collections import defaultdict
from uuid import uuid1

from loguru import logger


def list_dir(
        cur_path: str,
        ext_filter: Optional[List] = None
) -> Union[Dict, List]:
    """
    列举文件根目录下各文件的路径.
    Parameters
    ----------
    cur_path: str
        根目录.
    ext_filter: Optional[List], default None
        用作筛选的后缀名.

    Returns
    -------
    Union[Dict, List]
        文件路径列表.
    """
    file_paths = []
    for root, dirs, files in os.walk(cur_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    if ext_filter is not None:
        result_dict = {x: [] for x in ext_filter}
        for x in file_paths:
            cur_ext = os.path.splitext(x)[-1].lower()
            if cur_ext in ext_filter:
                result_dict[cur_ext].append(x)

        return result_dict

    return file_paths


# In[4]:


def data_persistence(
        folder: str,
        file_name: str,
        info: Union[str, Dict],
        encoding_way: str = 'utf-8',
        mode: str = 'json',
        indent: int = 0
) -> True:
    """
    保存json、txt数据到文件.

    Parameters
    ----------
    folder: str
        文件夹名.
    file_name: str
        文件名.
    info: Union[str, Dict]
        需要保存的文件，文本或字典.
    encoding_way: str, default `utf-8`
        编码方式.
    mode: str, default `JSON`
        保存模式.
    indent: int
        缩进空格数.

    Returns
    -------
    True
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, file_name), mode='w', encoding=encoding_way) as f:
        if mode == 'json':
            json.dump(info, f, ensure_ascii=False, indent=indent)
        else:
            f.write(info)
    return True


def move_file(
        src: str,
        dst: str
) -> str:
    """
    复制目标文件到指定位置, 如果没有文件夹会自动创建.

    Parameters
    ----------
    src: str
        文件原路径.
    dst: str
        文件新路径.
    Returns
    -------
    str
        文件新路径.
    """
    output_folder, f_name = os.path.split(dst)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return move(src, dst)


def list_to_dict(
        raw_list,
        func
):
    result = defaultdict(list)
    for x in raw_list:
        k = func(x)
        if k not in result:
            result[k] = []
        result[k].append(x)

    return result


# In[21]:


class CocoParser:
    def __init__(
            self,
            data: Dict
    ):
        self._images = data.get('images', [])
        self._annotations = data.get('annotations', [])
        self._categories = data.get('categories', [])
        self.image_map = self._parse_images()
        self.category_map = self._parse_categories()
        self.annotation_map = self._parse_annotations()

    def _parse_images(
            self
    ):
        return {
            x['id']: x['file_name']
            for x in self._images
        }

    def _parse_categories(
            self
    ):
        cat = {
            x['id']: {
                'name': x['name'],
                'keypoints': {i + 1: kp for i, kp in enumerate(x.get('keypoints', []))},
                'skeleton': x.get('skeleton', [])
            }
            for x in self._categories
        }

        for k, v in cat.items():
            v['skeleton'] = [{'end': end, 'start': start} for end, start in v['skeleton']]
            cat[k] = v

        return cat

    def _parse_annotations(
            self
    ):
        anns = filter(
            lambda x: set(x.keys()).intersection({'segmentation', 'bbox', 'keypoints'}) != set(),
            self._annotations
        )

        return list_to_dict(
            anns,
            func=lambda x: x['image_id']
        )

    @staticmethod
    def coco_polygon_to_basic(
            polygon: List
    ) -> List:
        return [
            {
                'x': polygon[2 * i],
                'y': polygon[2 * i + 1]
            }
            for i in range(len(polygon) // 2)
        ]

    @staticmethod
    def coco_bbox_to_basic(
            bbox: List
    ) -> List:
        xmin, ymin, w, h = bbox
        xmax, ymax = xmin + w, ymin + h

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
            }
        ]


class OntoParser:
    def __init__(
            self,
            classes,
            classifications
    ):
        self.classes = self._parse_classes(classes)
        self.skeleton = self._parse_skeleton(classes)

    @staticmethod
    def _parse_classes(
            classes
    ):
        return {(x['toolType'], x['name']): x['id'] for x in classes}

    @staticmethod
    def _parse_skeleton(
            classes
    ):
        sk = None
        for x in classes:
            if x['toolType'] == 'SKELETON':
                sk = x
                break

        if sk is None:
            return {
                'nodes': {},
                'node_ids': {},
                'node_attr': {},
                'lines': []
            }

        # 解析node uuid
        sk_cfg = sk['toolTypeOptions']['skeletonConfig']
        uuid_dict = {x['uuid']: (i + 1) for i, x in enumerate(sk_cfg['pointList'])}

        # 解析node 属性
        attr_dict = {x['attribute']: (x['id'], x['color']) for x in sk_cfg['tagList']}

        # 解析连线
        lines_list = []
        for x in sk_cfg['lineList']:
            line1, line2 = x['relationIds']
            lines_list.append(
                {
                    'start': uuid_dict[line1],
                    'end': uuid_dict[line2]
                }
            )

        return {
            'nodes': uuid_dict,
            'node_ids': {v: k for k, v in uuid_dict.items()},
            'node_attr': attr_dict,
            'lines': lines_list
        }


def trans(input_json):
    code = 'OK'
    message = ''

    origin_path = input_json.get('originPath')
    target_path = input_json.get('targetPath')

    start_time = time.time()

    # 解析ontology
    clss = input_json.get('datasetClassList')
    clfs = input_json.get('datasetClassificationList')
    op = OntoParser(clss, clfs)

    logger.info(f'time(s): {time.time() - start_time:.3f}; '
                f'script: upload_model/common/image/coco; '
                f'info: parse ontology finished')

    # 将图片路径读上来
    imgs = list_dir(origin_path, ['.jpg', '.png', '.jpeg'])
    for k, v in imgs.items():
        imgs[k] = {os.path.basename(img): img for img in v}

    logger.info(f'time(s): {time.time() - start_time:.3f}; '
                f'script: upload_model/common/image/coco; '
                f'info: parse images finished')

    # 讲json读上来并遍历标注结果
    jsons = list_dir(origin_path, ['.json'])['.json']
    for j in jsons[:]:
        logger.info(f'time(s): {time.time() - start_time:.3f}; '
                    f'script: upload_model/common/image/coco; '
                    f'info: parsing {j}')

        rel_path = os.path.relpath(j, origin_path)
        root_folder = Path(rel_path).parts[:-2]

        cp = CocoParser(json.load(open(j, encoding='utf-8')))

        for k, v in list(cp.annotation_map.items())[:]:
            # 读取并复制图片
            cur_img_name = cp.image_map.get(k)
            if not cur_img_name:
                message = f"Miss image files: image_id-{k}..."
                logger.info(f'time(s): {time.time() - start_time:.3f}; '
                            f'script: upload_model/common/image/coco; '
                            f'info: miss {cur_img_name}')
                continue

            raw_name, cur_img_ext = os.path.splitext(cur_img_name)
            cur_img_ext = cur_img_ext.lower()
            if cur_img_name in imgs[cur_img_ext]:
                org_path = imgs[cur_img_ext][cur_img_name]
                new_path = os.path.join(target_path, *root_folder, cur_img_name)
                move_file(
                    org_path,
                    new_path
                )

                logger.info(f'time(s): {time.time() - start_time:.3f}; '
                            f'script: upload_model/common/image/coco; '
                            f'info: copy {cur_img_name} finished')

            else:
                message = f"Miss image files: {cur_img_name}..."
                logger.info(f'time(s): {time.time() - start_time:.3f}; '
                            f'script: upload_model/common/image/coco; '
                            f'info: miss {cur_img_name}')
                continue

            # 反显json
            cur_result = {
                'instances': []
            }

            for single_ann in v:
                # 不支持is_crowd == 1
                if single_ann.get('iscrowd', 0) == 1:
                    continue

                # 获取category
                cur_cat = cp.category_map.get(single_ann.get('category_id')) or {"name": "noclass"}

                # 判断是否编组
                if 'segmentation' in single_ann and 'keypoints' in single_ann and single_ann.get('num_keypoints',
                                                                                                 0) != 0:
                    group_id = str(uuid1())
                elif 'segmentation' in single_ann and len(single_ann['segmentation']) > 1:
                    group_id = str(uuid1())
                else:
                    group_id = None

                # 反显矩形框
                if "bbox" in single_ann and "segmentation" not in single_ann:
                    xmin, ymin, w, h = single_ann["bbox"]

                    cur_ann = {
                        'type': 'BOUNDING_BOX',
                        'contour': {
                            'points': [
                                {
                                    "x": xmin,
                                    "y": ymin
                                },
                                {
                                    "x": xmin + w,
                                    "y": ymin
                                },
                                {
                                    "x": xmin + w,
                                    "y": ymin + h
                                },
                                {
                                    "x": xmin,
                                    "y": ymin + h
                                }
                            ]
                        },
                        'className': cur_cat['name'],
                        'groups': [group_id],
                    }

                    cur_ann['classId'] = op.classes.get((cur_ann['type'], cur_ann['className']))

                    cur_result['instances'].append(cur_ann)

                # 反显多边形
                if 'segmentation' in single_ann:
                    for single_polygon in single_ann['segmentation']:
                        cur_ann = {
                            'type': 'POLYGON',
                            'contour': {
                                'points': cp.coco_polygon_to_basic(single_polygon)
                            },
                            'className': cur_cat['name'],
                            'groups': [group_id],
                        }

                        cur_ann['classId'] = op.classes.get((cur_ann['type'], cur_ann['className']))

                        cur_result['instances'].append(cur_ann)

                # 反显骨骼点
                if 'keypoints' in single_ann and single_ann['num_keypoints'] != 0:
                    cur_ann = {
                        'type': 'SKELETON',
                        'contour': {
                            'lines': op.skeleton['lines'] or cur_cat['skeleton'],
                            'nodes': []
                        },
                        'className': cur_cat['name'],
                        'groups': [group_id],
                    }

                    cur_ann['classId'] = op.classes.get((cur_ann['type'], cur_ann['className']))

                    # 单独处理每个骨骼点
                    for node_i in range(len(single_ann['keypoints']) // 3):
                        x = single_ann['keypoints'][3 * node_i]
                        y = single_ann['keypoints'][3 * node_i + 1]
                        v = single_ann['keypoints'][3 * node_i + 2]
                        node_cat = cur_cat['keypoints'].get(single_ann['keypoints'][3 * node_i + 2], False)
                        node_attr_id, node_attr_color = op.skeleton['node_attr'].get(str(v), (None, None))
                        cur_node = {
                            'attr': {
                                'code': str(v),
                                'color': node_attr_color,
                                'id': node_attr_id,
                                'valid': bool(node_cat)
                            },
                            'position': {
                                'x': x,
                                'y': y
                            },
                        }

                        cur_ann['contour']['nodes'].append(cur_node)

                    cur_result['instances'].append(cur_ann)

                # 编组
                if group_id:
                    cur_ann = {
                        'id': group_id,
                        'type': 'GROUP',
                        'contour': {},
                        'className': 'group',
                        'groups': [],
                    }
                    cur_ann['classId'] = op.classes.get((cur_ann['type'], cur_ann['className']))
                    cur_result['instances'].append(cur_ann)

                # 日志
                logger.info(f'time(s): {time.time() - start_time:.3f}; '
                            f'script: upload_model/common/image/coco; '
                            f'info: current annotation - {single_ann["id"]}')

            # 全部处理完了保存
            data_persistence(
                os.path.join(target_path, *root_folder, 'result'),
                raw_name + '.json',
                cur_result
            )

    return code, message
