import yaml

from backend_algorithms.export_model import ImageExportExecutor
from backend_algorithms.utils.image import basic_bbox_to_yolo

CODE_MAP = {
    "common": 2.0,
    "cover": 1.0,
    "outside": 0.0
}


def convert(
        single_data,
        class_map
):
    iw = single_data.iw
    ih = single_data.ih

    ann_result = single_data.get_results()
    new_result = []
    for y in ann_result.instances:
        class_number = class_map.get(y.class_id, -1)

        if y.type == "BOUNDING_BOX":
            new_result.append(
                ' '.join(
                    map(
                        str,
                        [
                            class_number,
                            *basic_bbox_to_yolo(y.points, iw, ih)
                        ]
                    )
                )
            )

        elif y.type == "POLYGON":
            cur_row = [class_number]
            for p in y.points:
                cur_row.extend([p['x'] / iw, p['y'] / ih])
            new_result.append(
                ' '.join(map(str, cur_row))
            )

        elif y.type == "SKELETON":
            cur_row = [class_number]
            points = [p["position"] for p in y.nodes]
            cur_row.extend(basic_bbox_to_yolo(points, iw=iw, ih=ih))
            for p in y.nodes:
                org_code = (p["attr"]["code"] or '').lower()
                cur_node = [
                    p["position"]['x'] / iw,
                    p["position"]['y'] / ih,
                    CODE_MAP.get(org_code, org_code)
                ]
                if cur_node[-1] in ['0', 0.0]:
                    cur_node[0] = 0
                    cur_node[1] = 0
                cur_row.extend(
                    cur_node
                )
            new_result.append(
                ' '.join(map(str, cur_row))
            )

    return '\n'.join(new_result)


def trans(input_json):
    exe = ImageExportExecutor(input_json)

    classes = [x for x in exe.ontology_info.classes]
    class_map = exe.ontology_info.id_number_map
    if not class_map:
        class_map = {x.id: i for i, x in enumerate(classes)}

    exe.target_path.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(
        {
            'nc': len(class_map),
            'names': {x.number if isinstance(x.number, int) else i: x.name for i, x in enumerate(classes)}
        },
        (exe.target_path / 'data.yaml').open('w', encoding='utf-8'),
        allow_unicode=True
    )

    exe.convert_all_data(
        convert_func=convert,
        result_ext=".txt",
        class_map=class_map
    )

    exe.copy_original_files()
