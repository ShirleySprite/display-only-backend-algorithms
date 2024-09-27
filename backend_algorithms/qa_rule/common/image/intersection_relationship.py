from shapely.geometry import Polygon
import geopandas as gpd


def detect(input_json):
    data_ids = set()
    objects = []
    datas = input_json['data']
    for data in datas:
        for result in data['annotationResults']:
            data_id = result['dataId']

            # 检测多边形包含关系
            instances = [x for x in result['objects'] if x['type'] == 'POLYGON']

            # 构造polygon对象
            polygons = []
            for x in instances:
                pts = x['contour']['points']
                holes = [[[p['x'], p['y']] for p in intr['points'] if p != {}] for intr in
                         x['contour'].get('interior', [])]
                poly = Polygon([[p['x'], p['y']] for p in pts if p != {}], holes=holes)
                if not poly.is_valid:
                    continue

                # 生成polygon
                polygons.append([x['id'], poly])

            # 创建geodataframe
            gdf = gpd.GeoDataFrame(polygons, columns=['id', 'geo'], geometry='geo')

            # 检测相交
            for p in polygons:
                box_id = p[0]
                p1 = p[-1]

                inter_df = gdf.iloc[gdf.sindex.query(p1, predicate='intersects')]

                if len(inter_df) > 1:
                    data_ids.add(data_id)
                    objects.append({'objectId': box_id})

    rps_json = {
        "dataIds": list(data_ids),
        "classifications": [],
        "objects": objects
    }

    return rps_json
