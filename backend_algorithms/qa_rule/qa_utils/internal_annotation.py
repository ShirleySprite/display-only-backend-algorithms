from backend_algorithms.utils.annotation import ImageGroup, BBox, Polygon, Polyline, KeyPoint, Skeleton, Curve, \
    ImageCuboid, Circle, Ellipse, ImageSegment, LidarGroup, Lidar3DBox, Lidar3DLanePolygon, Lidar3DLanePolyline, Lidar2DRect, \
    Lidar2DBox, Lidar2DLanePolygon, Lidar2DLanePolyline, LidarSegment, Clip, Entity, Relation


class IntImageGroup(ImageGroup):
    ...


class IntBBox(BBox):
    ...


class IntPolygon(Polygon):
    ...


class IntPolyline(Polyline):
    ...


class IntKeyPoint(KeyPoint):
    ...


class IntSkeleton(Skeleton):
    ...


class IntCurve(Curve):
    ...


class IntImageCuboid(ImageCuboid):
    ...


class IntCircle(Circle):
    ...


class IntEllipse(Ellipse):
    ...


class IntImageSegment(ImageSegment):
    ...


class IntLidarGroup(LidarGroup):
    ...


class IntLidar3DBox(Lidar3DBox):
    ...


class IntLidar3DLanePolygon(Lidar3DLanePolygon):
    ...


class IntLidar3DLanePolyline(Lidar3DLanePolyline):
    ...


class IntLidar2DBox(Lidar2DBox):
    ...


class IntLidar2DRect(Lidar2DRect):
    ...


class IntLidar2DLanePolygon(Lidar2DLanePolygon):
    ...


class IntLidar2DLanePolyline(Lidar2DLanePolyline):
    ...


class IntLidarSegment(LidarSegment):
    ...


class IntClip(Clip):
    ...


class IntEntity(Entity):
    ...


class IntRelation(Relation):
    ...


image_map = {
    'BOUNDING_BOX': IntBBox,
    'POLYGON': IntPolygon,
    'POLYLINE': IntPolyline,
    'KEY_POINT': IntKeyPoint,
    'SKELETON': IntSkeleton,
    'CURVE': IntCurve,
    'GROUP': IntImageGroup,
    'IMAGE_CUBOID': IntImageCuboid,
    'CIRCLE': IntCircle,
    'ELLIPSE': IntEllipse,
    'MASK': IntImageSegment
}

lidar_map = {
    "GROUP": IntLidarGroup,
    '3D_BOX': IntLidar3DBox,
    '3D_LANE_POLYGON': IntLidar3DLanePolygon,
    '3D_LANE_POLYLINE': IntLidar3DLanePolyline,
    '2D_BOX': IntLidar2DBox,
    '2D_RECT': IntLidar2DRect,
    '2D_LANE_POLYGON': IntLidar2DLanePolygon,
    '2D_LANE_POLYLINE': IntLidar2DLanePolyline,
    "SEGMENTATION": IntLidarSegment
}

av_map = {
    'CLIP': IntClip
}

text_map = {
    "ENTITY": IntEntity,
    "RELATION": IntRelation
}
