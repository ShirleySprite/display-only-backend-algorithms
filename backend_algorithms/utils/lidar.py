import re
import struct
import math
from itertools import product
from pathlib import Path
from typing import List, Union, Dict, Tuple

import numpy as np
import lzf
from numpy.lib.recfunctions import repack_fields
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point, MultiPoint

numpy_pcd_type_mappings = [
    (np.dtype('float32'), ('F', 4)),
    (np.dtype('float64'), ('F', 8)),
    (np.dtype('uint8'), ('U', 1)),
    (np.dtype('uint16'), ('U', 2)),
    (np.dtype('uint32'), ('U', 4)),
    (np.dtype('uint64'), ('U', 8)),
    (np.dtype('int16'), ('I', 2)),
    (np.dtype('int32'), ('I', 4)),
    (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z i
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA binary
"""


def transform_matrix(
        t_vec: Union[np.ndarray, List, Tuple],
        rot_mat: Union[np.ndarray, List]
) -> np.ndarray:
    return np.vstack(
        [
            np.hstack(
                [
                    np.array(rot_mat),
                    np.array(t_vec).reshape(-1, 1)
                ]
            ),
            [0, 0, 0, 1]
        ]
    )


def to_basic_ext(
        t_vec: List,
        rot_mat: Union[np.ndarray, List],
        inv: bool = False
) -> List:
    trans = transform_matrix(t_vec, rot_mat)

    if inv:
        trans = np.linalg.inv(trans)

    return trans.ravel().tolist()


def to_basic_int(
        intrinsic: Union[np.ndarray, List]
) -> Dict:
    if isinstance(intrinsic, List):
        intrinsic = np.array(intrinsic)
    intrinsic = intrinsic.reshape(3, 3)

    return {
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2])
    }


def alpha_in_pi(a):
    pi = math.pi
    return a - math.floor((a + pi) / (2 * pi)) * 2 * pi


def count_points(
        pc: np.ndarray,
        cx: float,
        cy: float,
        cz: float,
        dx: float,
        dy: float,
        dz: float,
        rx: float,
        ry: float,
        rz: float
) -> int:
    inv_rot_mat = R.from_euler('XYZ', [rx, ry, rz]).as_matrix().T
    pc_rotated = (pc[:, :3] - np.array([cx, cy, cz])) @ inv_rot_mat.T
    dx, dy, dz = dx / 2, dy / 2, dz / 2
    mask = (
            (-dx < pc_rotated[:, 0]) & (pc_rotated[:, 0] < dx) &  # x
            (-dy < pc_rotated[:, 1]) & (pc_rotated[:, 1] < dy) &  # y
            (-dz < pc_rotated[:, 2]) & (pc_rotated[:, 2] < dz)  # z
    )

    return len(pc[mask])


def get_pose(
        cx,
        cy,
        cz,
        rx,
        ry,
        rz
) -> np.ndarray:
    return transform_matrix(
        t_vec=[cx, cy, cz],
        rot_mat=R.from_euler('XYZ', [rx, ry, rz]).as_matrix()
    )


def get_corners(
        dx,
        dy,
        dz,
        pose
) -> np.ndarray:
    arr = np.array([-0.5, 0.5])
    box_range = (
        np.array([x * arr for x in [dx, dy, dz]])
    )
    f_dim_box = np.insert(
        np.array(list(product(*box_range))),
        3,
        1,
        axis=1
    )

    return (pose @ f_dim_box.T)[:3, :].T


def get_distance(
        corners
) -> Tuple:
    xy_corners = corners[:, :2]
    proj_area = MultiPoint(xy_corners).convex_hull
    point_o = Point([0, 0])

    return proj_area.distance(point_o), proj_area.hausdorff_distance(point_o)


class PointCloud:
    def __init__(self, pcd_file, valid_points=True):
        self.metadata = None
        self.code = None
        self.invalid_points = 0

        if pcd_file is not None:
            if isinstance(pcd_file, (str, Path)):
                with open(pcd_file, 'rb') as f:
                    self.data = self._load_from_file(f)
            else:
                self.data = self._load_from_file(pcd_file)

        if valid_points:
            self.validate_points()

    @property
    def fields(self):
        return self.data.dtype.names

    def validate_points(self):
        pc = self.numpy(fields=['x', 'y', 'z'])
        mask = ~np.isnan(pc).any(axis=1) & (pc != 0).any(axis=1)
        self.invalid_points = len(pc) - mask.sum()
        if self.invalid_points > 0:
            self.data = self.data[mask]

    def valid_fields(self, fields=None):
        if fields is None:
            fields = self.data.dtype.names
        else:
            fields = [
                f
                for f in fields
                if f in self.data.dtype.names
            ]
        return fields

    def numpy(self, fields=None, dtype=np.float32):
        fields = self.valid_fields(fields)
        return np.stack([
            self.data[name].astype(dtype)
            for name in fields
        ], axis=1)

    def normalized_fields(self, extra_fields: list = None):
        all_fields = set(self.fields)

        fields = ['x', 'y', 'z']
        for f in fields:
            if f not in all_fields:
                raise ValueError(f'can not find "{f}" field in pcd file')

        if 'intensity' in all_fields:
            fields.append('intensity')
        elif 'i' in all_fields:
            fields.append('i')

        if extra_fields:
            for f in extra_fields:
                if f in all_fields:
                    fields.append(f)
        return fields

    def normalized_numpy(self, extra_fields: list = None, dtype=np.float32):
        fields = self.normalized_fields(extra_fields)
        return self.numpy(fields, dtype)

    def normalized_pc(self, extra_fields: list = None):
        fields = self.normalized_fields(extra_fields)
        return repack_fields(self.data[fields])

    @staticmethod
    def _build_dtype(metadata):
        fieldnames = []
        typenames = []

        # process dulipcated names
        fields = metadata['fields']
        fields_dict = set()
        for i in range(len(fields)):
            name = fields[i]
            if name in fields_dict:
                while name in fields_dict:
                    name += '1'
                fields[i] = name
            fields_dict.add(name)

        for f, c, t, s in zip(fields,
                              metadata['count'],
                              metadata.get('type', 'F'),
                              metadata['size']):
            np_type = pcd_type_to_numpy_type[(t, s)]
            if c == 1:
                fieldnames.append(f)
                typenames.append(np_type)
            elif c == 0:  # zero count
                continue
            elif c < 0:  # negative count
                left_count = -c
                while left_count > 0:
                    left_count -= typenames[-1].itemsize
                    fieldnames.pop()
                    typenames.pop()
            else:
                fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
                typenames.extend([np_type] * c)
        dtype = np.dtype(list(zip(fieldnames, typenames)))
        return dtype

    def parse_header(self, lines):
        """ Parse header of PCD files.
        """
        metadata = {}
        for ln in lines:
            if ln.startswith('#') or len(ln) < 2:
                continue
            match = re.match('(\w+)\s+([\w\s\.\-]+)', ln)
            if not match:
                print("warning: can't understand line: %s" % ln)
                continue
            key, value = match.group(1).lower(), match.group(2)
            if key == 'version':
                metadata[key] = value
            elif key == 'fields':
                metadata[key] = value.lower().split()
            elif key == 'type':
                metadata[key] = value.split()
            elif key in ('size', 'count'):
                metadata[key] = list(map(int, value.split()))
            elif key in ('width', 'height', 'points'):
                metadata[key] = int(value)
            elif key == 'viewpoint':
                metadata[key] = map(float, value.split())
            elif key == 'data':
                metadata[key] = value.strip().lower()
        # add some reasonable defaults
        if 'count' not in metadata:
            metadata['count'] = [1] * len(metadata['fields'])
        if 'viewpoint' not in metadata:
            metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if 'version' not in metadata:
            metadata['version'] = '.7'
        return metadata

    @staticmethod
    def _parse_points_from_buf(buf, dtype):
        return np.frombuffer(buf, dtype=dtype)

    def parse_binary_compressed_pc_data(self, f, dtype):
        """ Parse lzf-compressed data.
        """
        fmt = 'II'
        compressed_size, uncompressed_size = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        compressed_data = f.read(compressed_size)

        buf = lzf.decompress(compressed_data, uncompressed_size)
        if len(buf) != uncompressed_size:
            raise IOError('Error decompressing data')
        # the data is stored field-by-field
        # pc_data = self._parse_points_from_buf(buf, dtype)
        num_points = uncompressed_size // dtype.itemsize
        pc_data = np.zeros(num_points, dtype=dtype)
        ix = 0
        for dti in range(len(dtype)):
            dt = dtype[dti]
            bytes = dt.itemsize * num_points
            column = np.fromstring(buf[ix:(ix + bytes)], dt)
            pc_data[dtype.names[dti]] = column
            ix += bytes
        return pc_data

    def _load_from_file(self, f):
        header = []
        for _ in range(11):
            ln = f.readline().decode("ascii").strip()
            header.append(ln)
            if ln.startswith('DATA'):
                metadata = self.parse_header(header)
                self.code = code = metadata['data']
                dtype = self._build_dtype(metadata)
                break
        else:
            raise ValueError("invalid file header")

        points = metadata['points']
        if code == 'ascii':
            if 'rgb' in dtype.names:
                pc = np.genfromtxt(f, dtype=dtype, delimiter=' ')  # np.loadtxt is too slow
            else:
                num_fields = len(dtype.fields)
                pc = np.fromstring(f.read(), dtype=np.float32, sep=' ', count=points * num_fields).reshape(-1,
                                                                                                           num_fields)
                pc = np.core.records.fromarrays([pc[:, i] for i in range(num_fields)], dtype=dtype)

            # pc = np.genfromtxt(f, dtype=dtype, delimiter=' ')
            # pc = np.fromfile(f, dtype=dtype, sep=' ', count=points) # error
        elif code == 'binary':
            rowstep = points * dtype.itemsize
            buf = f.read(rowstep)
            pc = self._parse_points_from_buf(buf, dtype)
        elif code == 'binary_compressed':
            pc = self.parse_binary_compressed_pc_data(f, dtype)
        else:
            raise ValueError(f'invalid pcd DATA: "{code}"')

        return pc

    @staticmethod
    def save_pcd(pc: np.ndarray, file):
        """
        :param structured ndarray
        :param file: str for file object
        """
        f = open(file, 'wb') if isinstance(file, (str, Path)) else file

        fields = pc.dtype.names
        dtypes = [pc.dtype[f] for f in fields]
        num_fields = len(fields)
        num_points = len(pc)
        headers = [
            '# .PCD v0.7 - Point Cloud Data file format',
            'VERSION 0.7',
            f'FIELDS {" ".join(fields)}',
            f'SIZE {" ".join([str(d.itemsize) for d in dtypes])}',
            f'TYPE {" ".join([d.kind.upper() for d in dtypes])}',
            f'COUNT {" ".join(["1"] * num_fields)}',
            f'WIDTH {num_points}',
            'HEIGHT 1',
            'VIEWPOINT 0 0 0 1 0 0 0',
            f'POINTS {num_points}',
            'DATA binary'
        ]
        header = bytes('\n'.join(headers) + '\n', 'ascii')
        f.write(header)
        f.write(pc.tobytes())

        if isinstance(file, str):
            f.close()
