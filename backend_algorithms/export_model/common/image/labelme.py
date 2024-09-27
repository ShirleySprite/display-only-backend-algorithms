from backend_algorithms.export_model import ImageExportExecutor
from backend_algorithms.utils.image import points_to_list


def convert(
        single_data
):
    type_map = {
        'POLYGON': 'polygon',
        'POLYLINE': 'linestrip',
        'CIRCLE': 'circle',
        'KEY_POINT': 'point',
        'BOUNDING_BOX': 'rectangle'
    }

    new_result = single_data.to_labelme()
    new_result["imagePath"] = single_data.org_path.name

    for inst in single_data.get_results().instances:
        ann_type = inst.type

        if ann_type not in type_map:
            continue

        points = points_to_list(inst.points)

        if ann_type == 'BOUNDING_BOX':
            xmin, ymin, xmax, ymax = inst.find_diagonal()
            points = [[xmin, ymin], [xmax, ymax]]

        elif ann_type == 'CIRCLE':
            ox, oy = inst.center_x, inst.center_y
            points = [[ox, oy], [ox + inst.radius, oy]]

        try:
            gid = inst.groups[0]
        except (TypeError, IndexError, KeyError):
            gid = None

        cur_ann = {
            'label': inst.class_name or inst.model_class,
            'points': points,
            'group_id': gid,
            'shape_type': type_map[ann_type],
            'flags': {}
        }

        new_result['shapes'].append(cur_ann)

    return new_result


def trans(input_json):
    ImageExportExecutor(input_json).convert_all_data(convert)
