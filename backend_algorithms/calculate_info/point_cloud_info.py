import numpy as np

from backend_algorithms.utils.lidar import PointCloud, count_points, get_pose, get_corners, get_distance
from backend_algorithms.utils.general import groupby


def cal_instances_info(
        instances
):
    results = []
    boxes = [x for x in instances if x["type"] == "3D_BOX"]
    for pcd_path, cur_boxes in groupby(
            boxes,
            func=lambda x: x["pcdFilePath"]
    ).items():
        pc_arr = PointCloud(pcd_path).numpy(fields=['x', 'y', 'z'])
        for obj in cur_boxes:
            contour = obj['contour']
            cx, cy, cz = contour['center3D']['x'], contour['center3D']['y'], contour['center3D']['z']
            dx, dy, dz = contour['size3D']['x'], contour['size3D']['y'], contour['size3D']['z']
            rx, ry, rz = contour['rotation3D']['x'], contour['rotation3D']['y'], contour['rotation3D']['z']
            points_n = count_points(
                pc_arr,
                cx,
                cy,
                cz,
                dx,
                dy,
                dz,
                rx,
                ry,
                rz
            )
            corners = get_corners(
                dx,
                dy,
                dz,
                get_pose(cx, cy, cz, rx, ry, rz)
            )
            min_d, max_d = get_distance(corners)
            obj_info = {
                "objectId": obj['id'],
                "pointN": points_n,
                'minDistance': min_d,
                'maxDistance': max_d,
                'minHeight': min(corners[:, 2]),
                'maxHeight': max(corners[:, 2])
            }
            results.append(obj_info)

    return results


def cal_segments_info(
        segments
):
    results = []
    for (pcd_path, seg_path), cur_segments in groupby(
            segments,
            func=lambda x: (x["pcdFilePath"], x["segmentResultFilePath"])
    ).items():
        label_pc = PointCloud(seg_path, valid_points=False).data['seg'].reshape(-1, 1)
        points = PointCloud(pcd_path).numpy(fields=['x', 'y', 'z'])
        if label_pc.shape[0] != points.shape[0]:
            continue

        labeled_pc = np.hstack((points, label_pc))

        for seg_obj in cur_segments:
            mask = (labeled_pc[:, -1] == seg_obj['no'])
            seg_obj_points = labeled_pc[mask]
            xy_norm = np.linalg.norm(seg_obj_points[:, :2], axis=1)
            seg_obj_info = {
                "objectId": seg_obj['id'],
                "pointN": int(np.sum(mask)),
                "minDistance": float(min(xy_norm)),
                "maxDistance": float(max(xy_norm)),
                "minHeight": float(min(seg_obj_points[:, 2])),
                "maxHeight": float(max(seg_obj_points[:, 2]))
            }
            results.append(seg_obj_info)

    return results


def cal_point_cloud_info(data):
    return cal_instances_info(data['instances']) + cal_segments_info(data['segments'])
