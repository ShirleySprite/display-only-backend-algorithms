from typing import Union, Optional, List, Dict, Callable

import numpy as np
import cv2  # noqa

from backend_algorithms.utils.general import drop_duplicates
from backend_algorithms.utils.lidar import PointCloud
from backend_algorithms.utils.annotation import Polygon
from backend_algorithms.utils.base_result import ImageBaseResult, LidarBaseResult, AVBaseResult, TextBaseResult
from backend_algorithms.export_model.export_utils.external_annotation import image_map, lidar_map, av_map, text_map


class ExternalResult:
    def __init__(
            self,
            result_list: List,
            source_type: Optional[Union[str, List]] = None
    ):
        if isinstance(source_type, str):
            source_type = [source_type]

        if source_type is not None:
            result_list = [x for x in result_list if x['sourceType'] in source_type]
        self._result = result_list

    def _prepare_classifications(
            self,
            key: Callable = lambda x: x["classificationId"]
    ):
        return drop_duplicates(
            dups=[y for x in self._result for y in (x.get('classifications') or [])],
            key=key
        )

    def _prepare_instances(
            self,
            key_name="instances"
    ):
        return [y for x in self._result for y in (x.get(key_name) or [])]


class ExternalImageResult(ExternalResult, ImageBaseResult):
    def __init__(
            self,
            result_list: List,
            source_type: Optional[Union[str, List]] = None
    ):
        ExternalResult.__init__(
            self,
            result_list=result_list,
            source_type=source_type
        )
        ImageBaseResult.__init__(
            self,
            classifications=self._prepare_classifications(),
            instances=self._prepare_instances(),
            segments=self._prepare_instances("segments"),
            instance_map=image_map
        )

    def gen_mask(
            self,
            img,
            color_map: Dict,
            instances: Optional[List] = None
    ) -> np.ndarray:
        fst_color = list(color_map.values())[0]
        if instances is None:
            instances = self.instances

        instances = sorted(filter(lambda x: x.type == 'POLYGON', instances), key=lambda x: x.area, reverse=True)

        inst: Polygon
        for inst in instances:
            cv2.fillPoly(
                img=img,
                pts=[
                    np.array([[c['x'], c['y']] for c in inst.points]).round().astype(np.int32),
                    *[
                        np.array([[i['x'], i['y']] for i in inter]).round().astype(np.int32)
                        for inter in inst.interior
                    ]
                ],
                color=color_map.get(inst.class_id, fst_color)
            )

        return img


class ExternalLidarResult(ExternalResult, LidarBaseResult):
    def __init__(
            self,
            result_list: List,
            segmentation: PointCloud,
            source_type: Optional[Union[str, List]] = None
    ):
        ExternalResult.__init__(
            self,
            result_list=result_list,
            source_type=source_type
        )
        LidarBaseResult.__init__(
            self,
            classifications=self._prepare_classifications(),
            instances=self._prepare_instances(),
            segments=self._prepare_instances(key_name="segments"),
            instance_map=lidar_map
        )
        self.segmentation = segmentation


class ExternalAVResult(ExternalResult, AVBaseResult):
    def __init__(
            self,
            result_list: List,
            source_type: Optional[Union[str, List]] = None
    ):
        ExternalResult.__init__(
            self,
            result_list=result_list,
            source_type=source_type
        )
        AVBaseResult.__init__(
            self,
            classifications=self._prepare_classifications(),
            instances=self._prepare_instances(),
            instance_map=av_map
        )


class ExternalTextResult(ExternalResult, TextBaseResult):
    def __init__(
            self,
            result_list: List,
            source_type: Optional[Union[str, List]] = None
    ):
        ExternalResult.__init__(
            self,
            result_list=result_list,
            source_type=source_type
        )
        TextBaseResult.__init__(
            self,
            classifications=self._prepare_classifications(),
            entities=self._prepare_instances(key_name="entities"),
            relations=self._prepare_instances(key_name="relations"),
            instance_map=text_map
        )
