import requests
import numpy as np
import os
from os.path import *
import json

from backend_algorithms.utils.general import groupby
from backend_algorithms.utils.lidar import alpha_in_pi


def load_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read()
        json_content = json.loads(content)
    return json_content


def list_result_files(in_path: str, match):
    file_list = []
    for root, _, files in os.walk(in_path):
        if basename(root) == 'result':
            for file in files:
                if os.path.splitext(file)[-1] == match:
                    file_list.append(os.path.join(root, file))
    return file_list


def list_data_files(in_path: str, match):
    file_list = []
    for root, _, files in os.walk(in_path):
        if basename(root) == 'data':
            for file in files:
                if os.path.splitext(file)[-1] == match:
                    file_list.append(os.path.join(root, file))
    return file_list


def gen_alpha(rz, ext_matrix, lidar_center):
    lidar_center = np.hstack((lidar_center, np.array([1])))
    cam_point = ext_matrix @ np.array([np.cos(rz), np.sin(rz), 0, 1])
    cam_point_0 = ext_matrix @ np.array([0, 0, 0, 1])
    ry = -1 * (alpha_in_pi(np.arctan2(cam_point[2] - cam_point_0[2], cam_point[0] - cam_point_0[0])))
    cam_center = ext_matrix @ lidar_center.T
    theta = alpha_in_pi(np.arctan2(cam_center[0], cam_center[2]))
    alpha = ry - theta
    return ry, alpha


def find_attr(data_list, target_key):
    attrs_map = {x['name']: x['value'] for x in data_list}
    return eval(attrs_map.get(target_key, '0'))


def ensure_dir(input_dir):
    if not exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
    return input_dir


def trans(input_json):
    message = ''
    errors = []

    origin_path = input_json.get('originPath')
    target_path = input_json.get('targetPath')

    for result_file in list_result_files(origin_path, '.json'):
        result_txt_path = dirname(result_file.replace(origin_path, target_path))
        result_contents = load_json(result_file)
        instances = []
        for result in result_contents:
            instances.extend(result['instances'])

        data_file = join(dirname(dirname(result_file)), 'data', basename(result_file))
        data_content = load_json(data_file)
        config_url = data_content.get('cameraConfig')
        if not config_url:
            error = f"The data named '{basename(result_file)}' lacks a camera parameter file and cannot be exported to kitti format"
            errors.append(error)
            continue
        else:
            cam_param = requests.get(config_url['url']).json()
            n_cams = len(cam_param)
            rects = [x for x in instances if x['type'] == '2D_RECT']
            rect_map = groupby(rects, func=lambda x: x["trackId"])
            obj_3d = {x['trackId']: x for x in instances if x['type'] == '3D_BOX'}

            # 补充2D结果
            for track_id, inst_3d in obj_3d.items():
                label = inst_3d["className"]
                cur_views = {x["contour"]["viewIndex"] for x in rect_map.get(track_id, [])}
                for view_id in range(n_cams):
                    if view_id in cur_views:
                        continue

                    add_rect = {
                        "type": "2D_RECT",
                        "trackId": track_id,
                        "classValues": [],
                        "contour": {
                            "points": [
                                {
                                    "x": 0,
                                    "y": 0
                                },
                                {
                                    "x": 0,
                                    "y": 0
                                }
                            ],
                            "viewIndex": view_id
                        },
                        "className": label
                    }
                    rects.append(add_rect)

            for rect in rects:
                cam_index = rect['contour']['viewIndex']
                ext_matrix = np.array(cam_param[cam_index]['camera_external']).reshape(4, 4)
                label = rect['className'] or "noclass"

                try:
                    truncated = find_attr(rect['classValues'], 'truncated')
                    occluded = find_attr(rect['classValues'], 'occluded')
                except Exception:
                    truncated = 0
                    occluded = 0
                x_list = []
                y_list = []
                for one_point in rect['contour']['points']:
                    x_list.append(one_point['x'])
                    y_list.append(one_point['y'])
                if rect['trackId'] in obj_3d.keys():
                    contour_3d = obj_3d[rect['trackId']]['contour']
                    length, width, height = contour_3d['size3D'].values()
                    cur_rz = contour_3d['rotation3D']['z']

                    ry, alpha = gen_alpha(cur_rz, ext_matrix,
                                          np.array(list(contour_3d['center3D'].values())))

                    point = list(contour_3d['center3D'].values())
                    temp = np.hstack((np.array([point[0], point[1], point[2] - height / 2]), np.array([1])))
                    x, y, z = list((ext_matrix @ temp))[:3]
                    score = 1
                    string = f"{label} {truncated:.2f} {occluded} {alpha:.2f} " \
                             f"{min(x_list):.2f} {min(y_list):.2f} {max(x_list):.2f} {max(y_list):.2f} " \
                             f"{height:.2f} {width:.2f} {length:.2f} " \
                             f"{x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score}\n"
                else:
                    string = f"DontCare -1 -1 -10 " \
                             f"{min(x_list):.2f} {min(y_list):.2f} {max(x_list):.2f} {max(y_list):.2f} " \
                             f"-1 -1 -1 -1000 -1000 -1000 -10\n"

                result_txt_file = join(result_txt_path, f"label_{cam_index}",
                                       splitext(basename(result_file))[0] + '.txt')
                ensure_dir(dirname(result_txt_file))
                with open(result_txt_file, 'a+', encoding='utf-8') as tf:
                    tf.write(string)

    if errors:
        check_file = join(target_path, 'check_info.txt')
        with open(check_file, 'w', encoding='utf-8') as cf:
            cf.write('\n'.join(errors))
    code = 'OK' if not message else 'ERROR'

    return code, message
