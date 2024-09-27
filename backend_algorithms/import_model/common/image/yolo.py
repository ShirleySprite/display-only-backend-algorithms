import os
import csv
from pathlib import Path
from shutil import copyfile, move
from typing import List, Dict, Union, Optional
from collections import defaultdict

import ujson
import yaml
from PIL import Image


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
) -> defaultdict:
    result = defaultdict(list)
    for x in raw_list:
        k = func(x)
        if k not in result:
            result[k] = []
        result[k].append(x)

    return result


def gen_name_id_mapping(class_list, tool_type):
    mapping = {x['name']: x['id'] for x in class_list if x['toolType'] == tool_type}
    return mapping


class YoloParser:
    def __init__(
            self,
            dataset_path: str
    ):
        self.imgs = self._get_imgs(dataset_path)
        self.ann_data = self._get_txts(dataset_path)
        self.onto_dict = self._parse_yaml(dataset_path)

    @staticmethod
    def _get_imgs(
            dataset_path
    ):
        imgs = list_dir(dataset_path, ['.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp'])
        imgs = [i for v in imgs.values() for i in v]
        result = list_to_dict(
            imgs,
            func=lambda x: Path(x).parts[:-2]
        )

        for batch, cur_imgs in result.items():
            result[batch] = {
                os.path.splitext(os.path.basename(img))[0]: img
                for img in cur_imgs
            }

        return result

    @staticmethod
    def _get_txts(
            dataset_path
    ):
        txts = list_dir(dataset_path, ['.txt'])['.txt']

        return list_to_dict(
            filter(lambda x: Path(x).parts[-2] == 'labels', txts),
            func=lambda x: Path(x).parts[:-2]
        )

    @staticmethod
    def _parse_yaml(
            dataset_path
    ):
        yaml_file = list_dir(dataset_path, ['.yaml'])['.yaml'][0]
        data = yaml.load(open(yaml_file, encoding='utf-8'), Loader=yaml.FullLoader)

        return {i: x for i, x in enumerate(data['names'])}

    @staticmethod
    def yolo_bbox_to_basic(
            bbox: List,
            iw,
            ih
    ) -> List:
        xc, yc, w, h = map(float, bbox)
        xmin = (xc - w / 2) * iw
        xmax = (xc + w / 2) * iw
        ymin = (yc - h / 2) * ih
        ymax = (yc + h / 2) * ih

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


def trans(input_json):
    code = 'OK'
    message = ''

    origin_path = input_json.get('originPath')
    target_path = input_json.get('targetPath')
    class_mapping = gen_name_id_mapping(input_json.get('datasetClassList'), 'BOUNDING_BOX')

    try:
        yp = YoloParser(origin_path)
    except (KeyError, IndexError):
        code = "ERROR"
        message = "Incorrect file format. Please check our instruction document."

        return code, message

    for batch_folder, txts in yp.ann_data.items():
        for txt in txts:
            raw_name = os.path.splitext(os.path.basename(txt))[0]

            # 定位图片
            try:
                old_img_path = yp.imgs[batch_folder][raw_name]
            except KeyError:
                message = f"Miss image files: {os.path.join(batch_folder[-1], raw_name)}..."
                continue

            path_parts = list(Path(old_img_path).parts)
            root_folder = os.path.join(*path_parts[:-2]).replace(origin_path, target_path)

            # 获取长宽a
            w, h = Image.open(old_img_path).size

            # 复制图片
            move_file(
                old_img_path,
                os.path.join(root_folder, path_parts[-1])
            )

            # 标注结果
            cur_result = {
                'instances': []
            }
            with open(txt, 'r') as file:
                csv_reader = csv.reader(file, delimiter=' ')
                for row in csv_reader:
                    class_name = yp.onto_dict[int(row[0])]
                    cur_ann = {
                        'type': 'BOUNDING_BOX',
                        'contour': {
                            'points': yp.yolo_bbox_to_basic(
                                row[1:],
                                w,
                                h
                            )
                        },
                        'className': class_name,
                        'classId': class_mapping.get(class_name),
                    }
                    cur_result['instances'].append(cur_ann)

            # 保存
            data_persistence(
                os.path.join(root_folder, 'result'),
                raw_name + '.json',
                cur_result
            )

    return code, message
