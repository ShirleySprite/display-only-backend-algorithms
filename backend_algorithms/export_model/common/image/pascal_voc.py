from xml.dom.minidom import parseString
from typing import List, Optional

import dicttoxml

from backend_algorithms.export_model import ImageExportExecutor


def voc_single_convert(
        x,
        instances: Optional[List] = None,
        is_round: bool = True
):
    header_dom = parseString(
        dicttoxml.dicttoxml(
            {
                'folder': str(x.org_path.parent),
                'filename': x.org_path.name,
                'source': {'database': 'Unknown'},
                'size': {'width': x.iw, 'height': x.ih, 'depth': 3},
                'segmented': 0
            },
            root=True,
            custom_root='annotation',
            attr_type=False
        )
    )

    objs = []
    if instances is None:
        instances = x.get_results().instances
    for y in instances:
        obj_type = y.type

        if is_round:
            y = y.round_contour()

        cur_obj = {
            'name': y.class_name or 'noclass',
            'pose': 'Unspecified',
            **y.simp_class_values
        }

        if obj_type == 'BOUNDING_BOX':
            cur_obj['bndbox'] = y.find_diagonal(
                keys=['xmin', 'ymin', 'xmax', 'ymax']
            )

        elif obj_type == 'POLYGON':
            poly = {}
            i = 1
            for p in y.points:
                poly.update({f'x{i}': p['x'], f'y{i}': p['y']})
                i += 1

            cur_obj['polygon'] = poly

        elif obj_type == 'POLYLINE':
            poly = {}
            i = 1
            for p in y.points:
                poly.update({f'x{i}': p['x'], f'y{i}': p['y']})
                i += 1

            cur_obj['line'] = poly

        elif obj_type == 'KEY_POINT':
            cur_obj['point'] = y.points[0]

        elif obj_type == 'CIRCLE':
            cur_obj['circle'] = {'cx': y.center_x, 'cy': y.center_y, 'r': y.radius}

        else:
            continue

        objs.append(cur_obj)

    objects_dom = parseString(
        dicttoxml.dicttoxml(
            objs,
            root=True,
            attr_type=False,
            item_func=lambda o: 'object'
        )
    )

    root_node = header_dom.getElementsByTagName('annotation')[0]
    for n in objects_dom.getElementsByTagName('object'):
        root_node.appendChild(n)

    return header_dom


def trans(input_json):
    exe = ImageExportExecutor(input_json)
    exe.convert_all_data(
        convert_func=voc_single_convert,
        result_ext=".xml"
    )
    exe.copy_original_files()
