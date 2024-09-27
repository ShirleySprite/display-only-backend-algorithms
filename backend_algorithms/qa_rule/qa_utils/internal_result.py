import time
from typing import Dict, List

from loguru import logger

from backend_algorithms.utils.base_result import ImageBaseResult, LidarBaseResult, AVBaseResult, TextBaseResult
from backend_algorithms.qa_rule.qa_utils.internal_annotation import image_map, lidar_map, av_map, text_map


class InternalResult:
    def __init__(
            self,
            data,
            **kwargs
    ):
        self.raw: Dict = data
        self.data_id: int = self.raw['dataId']
        self.validity: str = self.raw.get('validity') or 'INVALID'

        self.classifications = self.raw.get('classifications') or []

        self.instances = self.raw.get('objects') or []

        self._no = kwargs.get("no", -1)
        self._script = kwargs.get("script", "unknown")
        self._start_time = kwargs.get("start_time", 0)
        logger.info(f'time(s): {time.time() - self._start_time:.3f}; '
                    f'script: {self._script}; '
                    f'no: {self._no}; '
                    f'data_id: {self.data_id}')


class ImageInternalResult(InternalResult, ImageBaseResult):
    def __init__(
            self,
            data: Dict,
            **kwargs
    ):
        InternalResult.__init__(
            self,
            data=data,
            **kwargs
        )
        ImageBaseResult.__init__(
            self,
            classifications=self.classifications,
            instances=self.instances,
            segments=self.raw.get('segments') or [],
            instance_map=image_map
        )

        seg_info: List[Dict] = self.raw.get('segmentations')
        if isinstance(seg_info, list) and len(seg_info):
            self.seg_url = seg_info[0].get('segmentPointsFilePath')
        else:
            self.seg_url = None


class LidarInternalResult(InternalResult, LidarBaseResult):
    def __init__(
            self,
            data: Dict,
            **kwargs
    ):
        InternalResult.__init__(
            self,
            data=data,
            **kwargs
        )
        LidarBaseResult.__init__(
            self,
            classifications=self.classifications,
            instances=self.instances,
            segments=self.raw.get('segments') or [],
            instance_map=lidar_map
        )

        seg_info: List[Dict] = self.raw.get('segmentations')
        if isinstance(seg_info, list) and len(seg_info):
            self.seg_url = seg_info[0].get('segmentPointsFilePath')
        else:
            self.seg_url = None


class AVInternalResult(InternalResult, AVBaseResult):
    def __init__(
            self,
            data: Dict,
            **kwargs
    ):
        InternalResult.__init__(
            self,
            data=data,
            **kwargs
        )
        AVBaseResult.__init__(
            self,
            classifications=self.classifications,
            instances=self.instances,
            instance_map=av_map
        )


class TextInternalResult(InternalResult, TextBaseResult):
    def __init__(
            self,
            data: Dict,
            **kwargs
    ):
        InternalResult.__init__(
            self,
            data=data,
            **kwargs
        )
        TextBaseResult.__init__(
            self,
            classifications=self.classifications,
            entities=self.raw.get("entities") or [],
            relations=self.raw.get('relations') or [],
            instance_map=text_map
        )
