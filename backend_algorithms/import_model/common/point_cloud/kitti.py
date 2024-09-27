import re
import numpy as np
import math
import os
from os.path import *
import json
import shutil
from nanoid import generate
from numpy.linalg import inv
from loguru import logger


def trans(input_json):
    code = 'OK'
    message = ''
    origin_path = input_json.get('originPath')
    target_path = input_json.get('targetPath')
    class_mapping = gen_name_id_mapping(input_json.get('datasetClassList'), 'CUBOID')
    # 遍历目标路径下的所有数据，以最后一层级文件夹名称为指定名称来识别数据集目录
    data_paths = list_folders_path(origin_path, 'velodyne')
    if data_paths:
        for data_path in data_paths:
            dataset_path = dirname(data_path)
            target_save_path = dataset_path.replace(origin_path, target_path)
            kittidataset = KittiDataset(dataset_path, target_save_path, class_mapping)
            check_source = kittidataset.irregular_structure()
            if check_source:
                code = 'ERROR'
                message = '\n'.join(check_source)
            else:
                try:
                    kittidataset.import_dataset()
                    message = str(kittidataset.mc)
                    code = 'OK'
                except Exception as e:
                    code = 'ERROR'
                    logger.exception(e)
                    message = 'faild, unable to parse'
    else:
        message = 'The directory for data named velodyne was not found'
    return code, message


def ensure_dir(input_dir):
    if not exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
    return input_dir


def list_files(in_path: str, match):
    file_list = []
    for root, _, files in os.walk(in_path):
        for file in files:
            if os.path.splitext(file)[-1] == match:
                file_list.append(os.path.join(root, file))
    return file_list


def gen_name_id_mapping(class_list, tool_type):
    mapping = {x['name']: x['id'] for x in class_list if x['toolType'] == tool_type}
    return mapping


def list_folders_path(folder_path, folder_name):
    matching_folders = []
    for root, dirs, files in os.walk(folder_path):
        # 检查当前目录下是否存在指定名称的文件夹
        if folder_name in dirs:
            # 遍历当前目录下的子文件夹
            for dir_name in dirs:
                # 检查子文件夹是否为指定名称
                if dir_name == folder_name:
                    # 获取该文件夹的路径
                    folder_path = os.path.join(root, dir_name)
                    matching_folders.append(folder_path)

    return matching_folders


def move_file(src: str, dst: str):
    if not exists(src):
        return False
    else:
        ensure_dir(dirname(src))
        shutil.move(src, dst)
        return True


class MessageCollecter:
    def __init__(
            self
    ):
        self.total_info = {}

    def __call__(
            self,
            error_type,
            error_info
    ):
        self.total_info[error_type] = error_info

    def __str__(
            self
    ):
        message = []
        for k, v in self.total_info.items():
            message.append(f"{k}: {v}")

        return '\n'.join(message)


class KittiDataset:
    def __init__(self, dataset_dir, output_dir, class_mapping):
        self.mc = MessageCollecter()
        self.dataset_dir = dataset_dir
        self.calib_dir = join(dataset_dir, 'calib')
        self.image_dir = join(dataset_dir, 'image_2')
        self.label_dir = join(dataset_dir, 'label_2')
        self.velodyne_dir = join(dataset_dir, 'velodyne')
        self.class_mapping = class_mapping
        if not self.irregular_structure():
            self.pc_dir = ensure_dir(join(output_dir, 'lidar_point_cloud_0'))
            self.image0_dir = ensure_dir(join(output_dir, 'camera_image_0'))
            self.camera_config_dir = ensure_dir(join(output_dir, 'camera_config'))
            self.result_dir = ensure_dir(join(output_dir, 'result'))

    def irregular_structure(self):
        """校验数据集下的目录结构"""
        check_info = []
        for _dir in [self.calib_dir, self.image_dir, self.velodyne_dir]:
            if not exists(_dir):
                check_info.append(f"{_dir} is not exists")
        return check_info

    def import_dataset(self):
        """处理数据:以bin文件为基础，查找对应其他资源文件，若缺少相机参数文件，参数文件解析失败，结果解析失败等，此帧跳过"""
        for bin_file in list_files(self.velodyne_dir, '.bin'):
            file_name = splitext(basename(bin_file))[0]
            pcd_file = join(self.pc_dir, file_name + '.pcd')
            try:
                self.bin_to_pcd(bin_file, pcd_file)
            except Exception:
                self.mc('Some bin file cannot be parsed', f'e.g.:{relpath(bin_file, self.dataset_dir)}')

            calib_file = join(self.calib_dir, file_name + '.txt')
            if exists(calib_file):
                cfg_file = join(self.camera_config_dir, file_name + '.json')
                try:
                    cam_param = self.parse_cam_param(calib_file, cfg_file)
                except Exception:
                    self.mc('Some calib file cannot be parsed', f'e.g.:{relpath(calib_file, self.dataset_dir)}')
                    continue
            else:
                self.mc('calib file not found', f'e.g.:{relpath(bin_file, self.dataset_dir)}')
                continue

            label_file = join(self.label_dir, file_name + '.txt')
            result_file = join(self.result_dir, file_name + '.json')
            try:
                self.parse_result(label_file, cam_param['camera_external'], result_file)
            except Exception:
                self.mc('Some result cannot be parsed', f'e.g.:{relpath(label_file, self.dataset_dir)}')
                continue

            has_img = False
            for suffix in ['.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp']:
                img = join(self.image_dir, file_name + suffix)
                image0 = join(self.image0_dir, file_name + suffix)
                if exists(img):
                    move_file(img, image0)
                    has_img = True
                else:
                    continue
            if not has_img:
                self.mc('Some image not found', f'e.g.:{relpath(join(self.image_dir, file_name), self.dataset_dir)}')

    @staticmethod
    def alpha_in_pi(alpha):
        pi = math.pi
        return alpha - math.floor((alpha + pi) / (2 * pi)) * 2 * pi

    # 将点数据写入pcd文件(encoding=ascii)
    @staticmethod
    def bin_to_pcd(bin_file: str, pcd_file: str):
        try:
            with open(bin_file, 'rb') as bf:
                bin_data = bf.read()
                dtype = np.dtype([('x', 'float32'), ('y', 'float32'), ('z', 'float32'), ('i', 'float32')])
                points = np.frombuffer(bin_data, dtype=dtype)
                points = [list(o) for o in points]
                points = np.array(points)
                with open(pcd_file, 'wb') as pcd_file:
                    point_num = points.shape[0]
                    heads = [
                        '# .PCD v0.7 - Point Cloud Data file format',
                        'VERSION 0.7',
                        'FIELDS x y z i',
                        'SIZE 4 4 4 4',
                        'TYPE F F F U',
                        'COUNT 1 1 1 1',
                        f'WIDTH {point_num}',
                        'HEIGHT 1',
                        'VIEWPOINT 0 0 0 1 0 0 0',
                        f'POINTS {point_num}',
                        'DATA binary'
                    ]
                    header = '\n'.join(heads) + '\n'
                    header = bytes(header, 'ascii')
                    pcd_file.write(header)
                    pcd_file.write(points.tobytes())
        except Exception as e:
            print(f"{e}\n{bin_file} to {pcd_file} ===> failed : {bin_file} "
                  f"dtype isn't [('x','float32'),('y','float32'),('z','float32'),('i','float32')]")

    @staticmethod
    def parse_cam_param(calib_file, cfg_file):
        with open(calib_file) as f:
            for line in f.readlines():
                if line[:2] == "P2":
                    P2 = re.split(" ", line.strip())
                    P2 = np.array(P2[-12:], np.float32)
                    P2 = P2.reshape((3, 4))
                if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                    vtc_mat = re.split(" ", line.strip())
                    vtc_mat = np.array(vtc_mat[-12:], np.float32)
                    vtc_mat = vtc_mat.reshape((3, 4))
                    vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
                if line[:7] == "R0_rect" or line[:6] == "R_rect":
                    R0 = re.split(" ", line.strip())
                    R0 = np.array(R0[-9:], np.float32)
                    R0 = R0.reshape((3, 3))
                    R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                    R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
        vtc_mat = np.matmul(R0, vtc_mat)

        int_mat = P2[:, :3].ravel().tolist()

        cfg_data = {
            "camera_internal": {
                "fx": int_mat[0],
                "fy": int_mat[4],
                "cx": int_mat[2],
                "cy": int_mat[5]
            },
            "camera_external": vtc_mat.flatten().tolist()
        }
        with open(cfg_file, 'w', encoding='utf-8') as cf:
            json.dump([cfg_data], cf)
        return cfg_data

    def parse_result(self, label_file, cam_ext, result_file):
        ext_matrix = np.array(cam_ext).reshape(4, 4)
        if exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                objects = []
                num = 1
                for line in lines:
                    track_id = generate(size=16)
                    data = line.split(' ')
                    label = data[0]
                    x0, y0, x1, y1 = [float(x) for x in data[4:8]]
                    obj_rect = {
                        "type": "2D_RECT",
                        "className": label,
                        "classId": self.class_mapping.get(label),
                        "trackId": track_id,
                        "trackName": str(num),
                        "contour": {
                            "points": [{"x": x0, "y": y0}, {"x": x0, "y": y1},
                                       {"x": x1, "y": y1}, {"x": x1, "y": y0}],
                            "size3D": {"x": 0, "y": 0, "z": 0},
                            "center3D": {"x": 0, "y": 0, "z": 0},
                            "viewIndex": 0,
                            "rotation3D": {"x": 0, "y": 0, "z": 0}
                        }
                    }
                    objects.append(obj_rect)
                    if label not in ['DontCare', 'Misc']:
                        height, width, length = [float(x) for x in data[8:11]]
                        cam_center = [float(x) for x in data[11:14]]
                        lidar_center = inv(ext_matrix) @ np.hstack((cam_center, [1]))
                        ry = float(data[14])
                        cam_to_lidar_point = inv(ext_matrix) @ np.array([np.cos(-ry), 0, np.sin(-ry), 1])
                        point_0 = inv(ext_matrix) @ np.array([0, 0, 0, 1])
                        rz = np.arctan2(cam_to_lidar_point[1] - point_0[1], cam_to_lidar_point[0] - point_0[0])

                        obj = {
                            "type": "3D_BOX",
                            "className": label,
                            "classId": self.class_mapping.get(label),
                            "trackId": track_id,
                            "trackName": str(num),
                            "contour": {
                                "size3D": {
                                    "x": length,
                                    "y": width,
                                    "z": height
                                },
                                "center3D": {
                                    "x": lidar_center[0],
                                    "y": lidar_center[1],
                                    "z": lidar_center[2] + height / 2
                                },
                                "rotation3D": {
                                    "x": 0,
                                    "y": 0,
                                    "z": self.alpha_in_pi(rz)
                                }
                            }
                        }
                        objects.append(obj)
                    num += 1
                with open(result_file, 'w', encoding='utf-8') as rf:
                    json.dump({"instances": objects}, rf)
