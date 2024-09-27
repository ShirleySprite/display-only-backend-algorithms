import json
import time
from io import BytesIO
from base64 import b64encode
from pathlib import Path
from xml.dom.minidom import Document
from abc import abstractmethod
from typing import Union, Optional, List, Dict

import pandas as pd
import requests
from PIL import Image
from loguru import logger

from backend_algorithms.export_model.export_utils.external_result import ExternalResult, ExternalImageResult, \
    ExternalLidarResult, ExternalAVResult, ExternalTextResult
from backend_algorithms.utils.general import filter_parts
from backend_algorithms.utils.lidar import PointCloud


class ExportData:
    def __init__(
            self,
            meta_path,
            **kwargs
    ):
        self.meta_path: Path = Path(meta_path)
        self.meta: Dict = json.load(self.meta_path.open(encoding='utf-8'))
        self.data_id: int = self.meta['dataId']
        self.name: str = self.meta['name']
        self.org_path: Path = Path(self.name)

        self.result_path = self.meta_path.parent.parent / 'result' / self.meta_path.name

        self._logger = kwargs.get("logger", True)
        self._no = kwargs.get("no", -1)
        self._script = kwargs.get("script", "unknown")
        self._start_time = kwargs.get("start_time", 0)
        self._add_info = kwargs.get("add_info", '')

        if self._logger:
            self.log_self()

    def __str__(
            self
    ):
        return f"data_id: {self.data_id}; name: {self.name}"

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(meta_path=r'{self.meta_path}')"

    def log_self(
            self,
            add_info: Optional[str] = None,
            no: Optional[int] = None
    ):
        if no is None:
            no = self._no

        logger.info(
            f'time(s): {time.time() - self._start_time:.3f}; '
            f'script: {self._script}; '
            f'no: {no}; '
            f'data_id: {self.data_id}; '
            f'add_info: {self._add_info if add_info is None else add_info}'
        )

    def _find_file(
            self,
            folder_name
    ):
        file_path = self.meta_path.parent.parent / self.org_path.name
        if file_path.exists():
            return file_path

        file_path = file_path.parent / folder_name / file_path.name
        if file_path.exists():
            return file_path

    def prepare_path(
            self,
            target_path: Union[str, Path],
            file_ext: str = '.json'
    ) -> Path:
        output_path = target_path / self.org_path.with_suffix(file_ext)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def write_result(
            self,
            result: Union[List, Dict, Document, str, bytes, pd.DataFrame, Image.Image],
            target_path: Union[str, Path],
            file_ext: str = '.json',
            add_params: Optional[Dict] = None
    ):
        if add_params is None:
            add_params = {}

        output_path = self.prepare_path(
            target_path=target_path,
            file_ext=file_ext
        )

        if isinstance(result, (List, Dict)):
            json_params = {
                "ensure_ascii": False,
                **add_params
            }
            result = json.dumps(result, **json_params)

        elif isinstance(result, Document):
            result = result.toprettyxml()

        if isinstance(result, str):
            return output_path.write_text(result, encoding='utf-8')

        elif isinstance(result, bytes):
            return output_path.write_bytes(result)

        elif isinstance(result, pd.DataFrame):
            if file_ext == ".csv":
                return result.to_csv(output_path, **add_params)
            elif file_ext == ".xlsx":
                return result.to_excel(output_path, **add_params)
            else:
                raise ValueError('unsupported suffix')

        elif isinstance(result, Image.Image):
            return result.save(output_path, **add_params)

        raise ValueError('unexpected result type')

    def get_meta(
            self
    ):
        return requests.get(url=self.meta["meta"]["url"]).text

    @abstractmethod
    def get_results(
            self
    ) -> ExternalResult:
        pass

    def filter_parts(
            self,
            filters
    ):
        self.org_path = filter_parts(self.org_path, filters)


class ExportImageData(ExportData):
    def __init__(
            self,
            meta_path: Union[str, Path],
            **kwargs
    ):
        super().__init__(
            meta_path=meta_path,
            **kwargs
        )

        self._info = self.meta['images'][0]
        self.iw: int = self._info['width']
        self.ih: int = self._info['height']
        self.org_path: Path = Path(self._info['zipPath'] or self._info['filename'])

        self.file: Optional[Path] = self._find_file(folder_name="image_0")

        self.segmentation_file: Optional[Path] = self._find_segmentation_file()

    def _find_segmentation_file(
            self
    ):
        seg_path = self.result_path.parent / f'{self.name}_image_0_segmentation.png'
        if seg_path.exists():
            return seg_path

    def get_results(
            self,
            source_type: Optional[Union[str, List]] = None
    ) -> ExternalImageResult:
        if not self.result_path.exists():
            result = []
        else:
            result = json.load(self.result_path.open(encoding='utf-8'))

        return ExternalImageResult(
            result_list=result,
            source_type=source_type
        )

    def to_coco_image(
            self
    ):
        return {
            "id": self.data_id,
            "width": self.iw,
            "height": self.ih,
            "file_name": self.org_path.name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }

    def to_labelme(
            self,
            version: str = "5.2.1"
    ):
        img_data = b64encode(self.file.read_bytes()).decode(
            encoding="utf-8") if self.file else None

        return {
            "version": version,
            "flags": {},
            "shapes": [],
            "imagePath": str(self.org_path),
            "imageData": img_data,
            "imageHeight": self.ih,
            "imageWidth": self.iw
        }


class Export3DLidarData(ExportData):
    def __init__(
            self,
            meta_path: Union[str, Path],
            **kwargs
    ):
        super().__init__(
            meta_path=meta_path,
            **kwargs
        )

        self._info = self.meta['lidarPointClouds'][0]
        self.org_path: Path = Path(self._info['zipPath'] or self._info['filename'])

        self.segmentation_file: Optional[Path] = self._find_segmentation_file()

    def _find_segmentation_file(
            self
    ):
        seg_path = self.result_path.parent / f'{self.name}_lidar_point_cloud_0_segmentation.pcd'
        if seg_path.exists():
            return seg_path

    def _load_segmentation(
            self
    ):
        return PointCloud(self._find_segmentation_file(), valid_points=False)

    def get_results(
            self,
            source_type: Optional[Union[str, List]] = None
    ) -> ExternalLidarResult:
        if not self.result_path.exists():
            result = []
        else:
            result = json.load(self.result_path.open(encoding='utf-8'))

        return ExternalLidarResult(
            result_list=result,
            source_type=source_type,
            segmentation=self._load_segmentation()
        )

    def get_bin_pcd(
            self
    ) -> PointCloud:

        return PointCloud(BytesIO(requests.get(self._info.get('binaryUrl')).content))

    def get_org_pcd(
            self,
    ) -> PointCloud:

        return PointCloud(BytesIO(requests.get(self._info.get('url')).content))

    def get_camera_config(
            self
    ) -> Optional[List]:

        return requests.get((self.meta.get('cameraConfig') or {}).get('url')).json()


class Export4DLidarData(Export3DLidarData):
    def _find_segmentation_file(
            self
    ):
        seg_path = self.result_path.parent / f'{self.name}_lidar_point_cloud_segmentation.pcd'
        if seg_path.exists():
            return seg_path


class ExportAVData(ExportData):
    def __init__(
            self,
            meta_path: Union[str, Path],
            **kwargs
    ):
        super().__init__(
            meta_path=meta_path,
            **kwargs
        )

        self._info = self.meta['avs'][0]
        self.duration: float = self._info['duration'] or 0.0
        self.org_path: Path = Path(self._info['zipPath'] or self._info['filename'])

    def get_results(
            self,
            source_type: Optional[Union[str, List]] = None
    ) -> ExternalAVResult:
        if not self.result_path.exists():
            result = []
        else:
            result = json.load(self.result_path.open(encoding='utf-8'))

        return ExternalAVResult(
            result_list=result,
            source_type=source_type
        )


class ExportTextData(ExportData):
    def __init__(
            self,
            meta_path: Union[str, Path],
            **kwargs
    ):
        super().__init__(
            meta_path=meta_path,
            **kwargs
        )

        self._info = self.meta['texts'][0]
        self.org_path: Path = Path(self._info['zipPath'] or self._info['filename'])

        self.file: Optional[Path] = self._find_file(folder_name="text_0")

    def get_results(
            self,
            source_type: Optional[Union[str, List]] = None
    ) -> ExternalTextResult:
        if not self.result_path.exists():
            result = []
        else:
            result = json.load(self.result_path.open(encoding='utf-8'))

        return ExternalTextResult(
            result_list=result,
            source_type=source_type
        )
