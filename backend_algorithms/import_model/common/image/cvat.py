import os
from pathlib import Path
from shutil import move
from typing import List, Dict, Union, Optional
from collections import defaultdict
from xml.dom import minidom

import ujson


def list_dir(
        cur_path: str,
        ext_filter: Optional[List] = None
) -> Union[List, Dict]:
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
    Union[List, Dict]
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
            ujson.dump(info, f, ensure_ascii=False, indent=indent)
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


def gen_name_id_mapping(class_list):
    """根据工具类型，生成标签名字和ontology的映射关系"""
    mapping = {}
    for class_obj in class_list:
        tool_type = class_obj['toolType']
        if tool_type in mapping.keys():
            mapping[tool_type][class_obj['name']] = class_obj['id']
        else:
            mapping[tool_type] = {class_obj['name']: class_obj['id']}
    return mapping


class CVATParser:
    def __init__(
            self,
            xml_path: str
    ):
        self._data = minidom.parse(xml_path).documentElement
        self.is_video = bool(self._data.getElementsByTagName('track'))

        # 元数据，不动
        self.meta = self._parse_meta(self._data.getElementsByTagName('meta')[0])

        # 查找该xml同级的图片
        self.imgs = self._search_imgs(xml_path)
        if self.is_video:
            self.imgs = {str(i): v for i, v in enumerate(self.imgs.values())}

        # 解析标注数据
        if self.is_video:
            pre_parse = self._pre_parse_track()
        else:
            pre_parse = self._pre_parse_image()
        self.ann_data = self._parse_ann(pre_parse)

    @staticmethod
    def _search_imgs(
            xml_path
    ):
        # 写死图片文件夹为images
        img_folder = os.path.join(os.path.dirname(xml_path), 'images')
        result = {}
        for img in os.listdir(img_folder):
            result[img] = os.path.join(img_folder, img)

        return result

    @staticmethod
    def _parse_meta(
            meta
    ):
        return meta

    def _pre_parse_image(
            self
    ):
        ann_data = self._data.getElementsByTagName('image')
        total_data = {}
        for img in ann_data:
            img_path = self.imgs[img.getAttribute('name')]
            ann_list = []
            for x in img.childNodes:
                if isinstance(x, minidom.Element):
                    x_dict = dict(x.attributes.items())
                    x_dict['node_name'] = x.nodeName
                    ann_list.append(x_dict)
            ann_dict = list_to_dict(ann_list, func=lambda x: x['node_name'])
            total_data[img_path] = ann_dict

        return total_data

    def _pre_parse_track(
            self
    ):
        ann_data = self._data.getElementsByTagName('track')
        total_data = {k: [] for k in self.imgs.keys()}
        for track in ann_data:
            track_id = track.getAttribute('id')
            label = track.getAttribute('label')
            for x in track.childNodes:
                if isinstance(x, minidom.Element):
                    x_dict = dict(x.attributes.items())
                    cur_frame = x_dict['frame']
                    x_dict['trackId'] = track_id
                    x_dict['trackName'] = track_id
                    x_dict['label'] = label
                    x_dict['node_name'] = x.nodeName
                    total_data[cur_frame].append(x_dict)

        return {
            self.imgs[k]: list_to_dict(v, func=lambda x: x['node_name'])
            for k, v in total_data.items()
        }

    def _parse_ann(
            self,
            pre_parse
    ):
        trans_map = {
            'box': 'BOUNDING_BOX',
            'polygon': 'POLYGON',
            'polyline': 'POLYLINE',
            'cuboid': 'IMAGE_CUBOID',
            'points': 'KEY_POINT',
        }
        total_data = {}
        for img_path, ann_dict in pre_parse.items():
            ann_dict = {trans_map[k]: v for k, v in ann_dict.items() if k in trans_map}
            # 解析box
            for cur_box in ann_dict.get('BOUNDING_BOX', []):
                cur_box['points'] = self.cvat_bbox_to_basic(
                    cur_box['xtl'],
                    cur_box['ytl'],
                    cur_box['xbr'],
                    cur_box['ybr']
                )

            # 解析polygon
            for cur_polygon in ann_dict.get('POLYGON', []):
                cur_polygon['points'] = self.cvat_points_to_basic(
                    cur_polygon['points']
                )

            # 解析polyline
            for cur_polyline in ann_dict.get('POLYLINE', []):
                cur_polyline['points'] = self.cvat_points_to_basic(
                    cur_polyline['points']
                )

            # 解析points
            temp = []
            for cur_points in ann_dict.get('KEY_POINT', []):
                new_points = []
                label = cur_points['label']
                basic_pts = self.cvat_points_to_basic(
                    cur_points['points']
                )

                for pts in basic_pts:
                    new_points.append(
                        {
                            'label': label,
                            'points': [pts],
                            # 'group': gid
                        }
                    )

                temp.extend(new_points)

            if temp:
                ann_dict['KEY_POINT'] = temp

            # 解析cuboid
            for cur_cuboid in ann_dict.get('IMAGE_CUBOID', []):
                cur_cuboid['points'] = self.cvat_cuboid_to_basic(
                    cur_cuboid['xtl1'],
                    cur_cuboid['ytl1'],
                    cur_cuboid['xbr1'],
                    cur_cuboid['ybr1'],
                    cur_cuboid['xbr2'],
                    cur_cuboid['ytl2'],
                    cur_cuboid['xtl2'],
                    cur_cuboid['ybr2']
                )

            total_data[img_path] = ann_dict

        return total_data

    @staticmethod
    def cvat_points_to_basic(
            points: str
    ) -> List:

        result = []
        for p in points.split(';'):
            x, y = p.split(',')
            result.append(
                {
                    'x': float(x),
                    'y': float(y)
                }
            )

        return result

    @staticmethod
    def cvat_bbox_to_basic(
            xmin,
            ymin,
            xmax,
            ymax
    ) -> List:

        return [
            {
                'x': float(xmin),
                'y': float(ymin)
            },
            {
                'x': float(xmax),
                'y': float(ymin)
            },
            {
                'x': float(xmax),
                'y': float(ymax)
            },
            {
                'x': float(xmin),
                'y': float(ymax)
            }
        ]

    @staticmethod
    def cvat_cuboid_to_basic(
            xmin1,
            ymin1,
            xmax1,
            ymax1,
            xmin2,
            ymin2,
            xmax2,
            ymax2,
    ) -> List:

        return [
            {
                'x': float(xmin1),
                'y': float(ymin1)
            },
            {
                'x': float(xmax1),
                'y': float(ymin1)
            },
            {
                'x': float(xmax1),
                'y': float(ymax1)
            },
            {
                'x': float(xmin1),
                'y': float(ymax1)
            },
            {
                'x': float(xmin2),
                'y': float(ymin2)
            },
            {
                'x': float(xmax2),
                'y': float(ymin2)
            },
            {
                'x': float(xmax2),
                'y': float(ymax2)
            },
            {
                'x': float(xmin2),
                'y': float(ymax2)
            }
        ]


def trans(input_json):
    code = 'OK'
    message = ''

    origin_path = input_json.get('originPath')
    target_path = input_json.get('targetPath')
    class_mapping = gen_name_id_mapping(input_json.get('datasetClassList'))

    # 将xml读上来并遍历标注结果
    xmls = list_dir(origin_path, ['.xml'])['.xml']
    for xml_ in xmls[:]:
        cp = CVATParser(xml_)

        for img_old_path, ann_dict in cp.ann_data.items():
            # 读取并复制图片
            path_parts = list(Path(img_old_path).parts)
            img_name = path_parts[-1]
            root_folder = os.path.join(*path_parts[:-2]).replace(origin_path, target_path)
            move_file(
                img_old_path,
                os.path.join(root_folder, 'image_0', img_name)
            )

            # 反显json
            cur_result = {
                'instances': []
            }

            for ann_type, ann_objs in ann_dict.items():
                # 反显单个物体
                for single in ann_objs:
                    type_mapping = class_mapping.get(ann_type)
                    if type_mapping:
                        class_id = type_mapping.get(single['label'])
                    else:
                        class_id = None
                    cur_ann = {
                        'type': ann_type,
                        'contour': {
                            'points': single['points']
                        },
                        'className': single['label'],
                        'classId': class_id
                    }
                    if cp.is_video:
                        cur_ann['trackId'] = single['trackId']

                    cur_result['instances'].append(cur_ann)

            # 全部处理完了保存
            data_persistence(
                os.path.join(root_folder, 'result'),
                os.path.splitext(img_name)[0] + '.json',
                cur_result
            )

    return code, message
