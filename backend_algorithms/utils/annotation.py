from typing import Dict, Union, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon as SPolygon, Point, LineString

from backend_algorithms.utils.image import find_diagonal, points_to_list, round_points
from backend_algorithms.utils.lidar import get_pose, get_corners, alpha_in_pi
from backend_algorithms.utils.general import reg_dict, drop_duplicates, CustomTree, seconds_to_hms


class AnnotationObject:
    def __init__(
            self,
            ann: Dict
    ):
        self.raw: Dict = ann
        self._id = self.raw['id']
        self.type: str = self.raw['type']
        self.class_values: List[Dict] = self.raw.get('classValues') or []
        self.class_id: int = int(self.raw.get('classId') or -1)
        self.class_name: str = self.raw.get('className') or ''
        self.class_number: int = self.raw.get('classNumber') or -1
        self.model_class = self.raw.get("modelClass") or ''
        self.created_by = self.raw.get("createdBy") or ''

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(type: {self.type}; class_name: {self.class_name})"

    def __hash__(
            self
    ):
        return hash(self.id)

    def __eq__(
            self,
            other
    ):
        if isinstance(other, self.__class__):
            return self.id == other.id
        return False

    @property
    def tree_class_values(
            self
    ) -> Optional[Dict]:
        tree = CustomTree()
        tree.create_node("Root", "root")
        for cv in self.class_values:
            tree.create_node(
                tag=cv["name"],
                identifier=cv["id"],
                parent=cv.get("pid") or "root",
                data=cv["value"]
            )

        return tree.to_dict(with_data=True)["Root"]

    @property
    def simp_class_values(
            self
    ) -> Dict:
        return {x['name']: x['value'] for x in self.class_values}

    @property
    def id(
            self
    ) -> str:
        return self._id


class ImageGroup(AnnotationObject):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            ann=instance
        )

        self.track_id: str = self.raw.get('trackId') or ''
        self.track_name: str = self.raw.get('trackName') or ''


class ImageInstance(AnnotationObject):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            ann=instance
        )

        self.track_id: str = self.raw.get('trackId') or ''
        self.track_name: str = self.raw.get('trackName') or ''

        self._contour: Dict = self.raw.get('contour') or {}
        self.points: List[Dict] = self._process_points()

        self.groups: List[str] = self.raw.get('groups') or []

    def _process_points(
            self
    ):
        points = self._contour.get('points') or []

        points = [
            reg_dict(x, keys=['x', 'y'])
            for x in points if x
        ]

        points = drop_duplicates(points, key=lambda x: (x['x'], x['y']))

        return points

    def find_siblings(
            self,
            instances: List
    ) -> List:
        siblings = []
        if not self.groups:
            return siblings

        for inst in instances:
            if inst.type == 'GROUP':
                continue

            if inst.id != self.id and inst.groups == self.groups:
                siblings.append(inst)

        return siblings

    def round_contour(
            self,
            round_param: Optional[int] = None
    ):
        new_contour = round_points(
            pts=self._contour,
            round_param=round_param
        )

        return self.__class__(
            {
                **self.raw,
                "contour": new_contour
            }
        )


class BBox(ImageInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.rotation: float = self._contour.get('rotation') or 0.0
        self.area: int = self._contour.get("area")
        if self.area is None:
            self.area = self.to_shape().area

    def find_diagonal(
            self,
            return_w_h: bool = False,
            keys: Optional[List] = None
    ) -> Union[List, Dict, None]:
        return find_diagonal(
            points_list=self.points,
            return_w_h=return_w_h,
            keys=keys
        )

    def points_to_list(
            self
    ) -> List:
        return points_to_list(
            pts=self.points
        )

    def to_shape(
            self
    ) -> SPolygon:
        return SPolygon(self.points_to_list())


class Polygon(ImageInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.interior: List[List[Dict]] = self._reg_interior()
        self.area: int = self._contour.get("area")
        if self.area is None:
            self.area = self.to_shape().area

    def _reg_interior(
            self
    ):
        org_interior = self._contour.get('interior') or []

        interior = []
        for hole in org_interior:
            cur_hole = []
            for p in (hole.get('points') or []):
                cur_hole.append(reg_dict(p, keys=['x', 'y']))

            if cur_hole:
                interior.append(cur_hole)

        return interior

    def find_diagonal(
            self,
            return_w_h: bool = False,
            keys: Optional[List] = None
    ) -> Union[List, Dict, None]:
        return find_diagonal(
            points_list=self.points,
            return_w_h=return_w_h,
            keys=keys
        )

    def points_to_list(
            self
    ) -> Tuple:
        return (
            points_to_list(
                pts=self.points
            ),
            points_to_list(
                pts=self.interior
            )
        )

    def to_shape(
            self
    ) -> SPolygon:
        shell, holes = self.points_to_list()

        return SPolygon(
            shell=shell,
            holes=holes
        )


class Polyline(ImageInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

    def points_to_list(
            self
    ) -> List:
        return points_to_list(
            pts=self.points
        )

    def to_shape(
            self
    ) -> LineString:
        return LineString(self.points_to_list())


class KeyPoint(ImageInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.x: Optional[float]
        self.y: Optional[float]
        self.x, self.y = self._get_xy()

    def _get_xy(
            self
    ):
        try:
            x = self.points[0]['x']
            y = self.points[0]['y']
        except (IndexError, KeyError):
            x = None
            y = None

        return x, y

    def to_shape(
            self
    ) -> Point:
        return Point([self.x, self.y])


class Skeleton(ImageInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.lines: List = self._contour.get('lines') or []
        self.nodes: List = self._contour.get('nodes') or []


class Curve(ImageInstance):
    ...


class ImageCuboid(ImageInstance):
    ...


class Circle(ImageInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.center_x: Optional[float]
        self.center_y: Optional[float]
        self.center_x, self.center_y = self._get_xy()

        self.radius = self._contour.get('radius') or 0.0

    def _get_xy(
            self
    ):
        try:
            center_x = self.points[0]['x']
            center_y = self.points[0]['y']
        except (IndexError, KeyError):
            center_x = None
            center_y = None

        return center_x, center_y

    def to_shape(
            self,
            resolution: int = 64
    ) -> SPolygon:
        return Point([self.center_x, self.center_y]).buffer(
            distance=self.radius,
            resolution=resolution
        )


class Ellipse(ImageInstance):
    ...


# image segment
class ImageSegment(AnnotationObject):
    def __init__(
            self,
            segment: Dict
    ):
        super().__init__(
            ann=segment
        )
        self.track_id: str = self.raw.get('trackId') or ''
        self.track_name: str = self.raw.get('trackName') or ''
        self._contour = self.raw.get('contour') or {}
        self.no: int = self.raw.get('no') or 0

        self.area: int = self._contour.get('area') or 0
        self.box: List[int] = self._contour.get('box') or [0, 0, 0, 0]
        self.mask_data: List[int] = self._contour.get('maskData') or []


class LidarInstance(AnnotationObject):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            ann=instance
        )

        self.track_id: str = self.raw.get('trackId') or ''
        self.track_name: str = self.raw.get('trackName') or ''
        self._contour = self.raw.get('contour') or {}

        self.groups: List[str] = self.raw.get('groups') or []

    def _process_points(
            self
    ):
        points = self._contour.get('points') or []

        points = [
            x
            for x in points if x
        ]

        points = drop_duplicates(
            points,
            key=lambda x: (x.get('x', 0), x.get('y', 0), x.get('z', 0))
        )

        return points


class LidarGroup(AnnotationObject):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            ann=instance
        )

        self.track_id: str = self.raw.get('trackId') or ''
        self.track_name: str = self.raw.get('trackName') or ''


class Lidar3DInstance(LidarInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.point_n: int = self._contour.get('pointN')
        self.multi_point_n: Dict = self._contour.get("multiPointN") or {}


class Lidar2DInstance(LidarInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.view_index: int = self._contour['viewIndex']
        self.points: List[Dict] = self._process_points()


class Lidar3DBox(Lidar3DInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.center: Dict = reg_dict(
            org_dict=self._contour.get('center3D') or {},
            keys=['x', 'y', 'z'],
            default_value=0
        )
        self.size: Dict = reg_dict(
            org_dict=self._contour.get('size3D') or {},
            keys=['x', 'y', 'z'],
            default_value=0
        )
        self.rotation: Dict = reg_dict(
            org_dict=self._contour.get('rotation3D') or {},
            keys=['x', 'y', 'z'],
            default_value=0
        )

    @property
    def affine_matrix(
            self
    ) -> np.ndarray:
        return get_pose(
            *self.center.values(),
            *self.rotation.values()
        )

    def get_corners(
            self
    ) -> np.ndarray:
        return get_corners(
            *self.size.values(),
            self.affine_matrix
        )

    def rotation_in_pi(
            self
    ):
        return {
            k: alpha_in_pi(v)
            for k, v in self.rotation.items()
        }


class Lidar3DLanePolygon(Lidar3DInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.points: List[Dict] = self._process_points()


class Lidar3DLanePolyline(Lidar3DInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.points: List[Dict] = self._process_points()

    def points_to_list(
            self
    ) -> List:
        return points_to_list(
            pts=self.points
        )

    @property
    def length(
            self
    ):
        polyline = np.array(self.points_to_list())

        return sum(
            np.linalg.norm(
                polyline[1:] - polyline[:-1],
                axis=1
            )
        )


class Lidar2DBox(Lidar2DInstance):
    ...


class Lidar2DRect(Lidar2DInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

    def find_diagonal(
            self,
            return_w_h: bool = False,
            keys: Optional[List] = None
    ) -> Union[List, Dict, None]:
        return find_diagonal(
            points_list=self.points,
            return_w_h=return_w_h,
            keys=keys
        )


class Lidar2DLanePolygon(Lidar2DInstance):
    ...


class Lidar2DLanePolyline(Lidar2DInstance):
    ...


# lidar segment
class LidarSegment(AnnotationObject):
    def __init__(
            self,
            segment: Dict
    ):
        super().__init__(
            ann=segment
        )
        self.track_id: str = self.raw.get('trackId') or ''
        self.track_name: str = self.raw.get('trackName') or ''
        self._contour = self.raw.get('contour') or {}
        self.no: int = self.raw.get('no') or 0

        self.point_n: int = self._contour.get('pointN') or 0


# av
class AVInstance(AnnotationObject):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            ann=instance
        )


class Clip(AVInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.start: float = self.raw.get('start') or 0.0
        self.end: float = self.raw.get('end') or 0.0
        self.length: float = self.end - self.start
        self.note: str = self.raw.get('note') or ''

    def to_hms(
            self
    ):
        return seconds_to_hms(self.start), seconds_to_hms(self.end)


class TextInstance(AnnotationObject):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            ann=instance
        )


class Entity(TextInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.start: int = self.raw.get('start') or 0
        self.end: int = self.raw.get('end') or 0
        self.length: int = self.raw.get('length') or (self.end - self.start)
        self.content = self.raw.get("content") or ''


class Relation(TextInstance):
    def __init__(
            self,
            instance: Dict
    ):
        super().__init__(
            instance=instance
        )

        self.source: str = self.raw.get("source") or ''
        self.target: str = self.raw.get("target") or ''
