from backend_algorithms.utils.annotation import ImageGroup, BBox, Polygon, Polyline, KeyPoint, Skeleton, Curve, \
    ImageCuboid, Circle, Ellipse, ImageSegment, LidarGroup, Lidar3DBox, Lidar3DLanePolygon, Lidar3DLanePolyline, Lidar2DRect, \
    Lidar2DBox, Lidar2DLanePolygon, Lidar2DLanePolyline, LidarSegment, Clip, Entity, Relation


class ExtImageGroup(ImageGroup):
    ...


class ExtBBox(BBox):
    ...


class ExtPolygon(Polygon):
    ...


class ExtPolyline(Polyline):
    ...


class ExtKeyPoint(KeyPoint):
    ...


class ExtSkeleton(Skeleton):
    ...


class ExtCurve(Curve):
    ...


class ExtImageCuboid(ImageCuboid):
    ...


class ExtCircle(Circle):
    ...


class ExtEllipse(Ellipse):
    ...


class ExtImageSegment(ImageSegment):
    ...


class ExtLidarGroup(LidarGroup):
    ...


class ExtLidar3DBox(Lidar3DBox):
    ...


class ExtLidar3DLanePolygon(Lidar3DLanePolygon):
    ...


class ExtLidar3DLanePolyline(Lidar3DLanePolyline):
    ...


class ExtLidar2DBox(Lidar2DBox):
    ...


class ExtLidar2DRect(Lidar2DRect):
    ...


class ExtLidar2DLanePolygon(Lidar2DLanePolygon):
    ...


class ExtLidar2DLanePolyline(Lidar2DLanePolyline):
    ...


class ExtLidarSegment(LidarSegment):
    ...


class ExtClip(Clip):
    ...


class ExtEntity(Entity):
    ...


class ExtRelation(Relation):
    ...


image_map = {
    'BOUNDING_BOX': ExtBBox,
    'POLYGON': ExtPolygon,
    'POLYLINE': ExtPolyline,
    'KEY_POINT': ExtKeyPoint,
    'SKELETON': ExtSkeleton,
    'CURVE': ExtCurve,
    'GROUP': ExtImageGroup,
    'IMAGE_CUBOID': ExtImageCuboid,
    'CIRCLE': ExtCircle,
    'ELLIPSE': ExtEllipse,
    "MASK": ExtImageSegment
}

lidar_map = {
    "GROUP": ExtLidarGroup,
    "3D_BOX": ExtLidar3DBox,
    "3D_LANE_POLYGON": ExtLidar3DLanePolygon,
    "3D_LANE_POLYLINE": ExtLidar3DLanePolyline,
    "2D_BOX": ExtLidar2DBox,
    "2D_RECT": ExtLidar2DRect,
    "2D_LANE_POLYGON": ExtLidar2DLanePolygon,
    "2D_LANE_POLYLINE": ExtLidar2DLanePolyline,
    "SEGMENTATION": ExtLidarSegment
}

av_map = {
    "CLIP": ExtClip
}

text_map = {
    "ENTITY": ExtEntity,
    "RELATION": ExtRelation
}
