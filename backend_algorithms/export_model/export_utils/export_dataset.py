import time
from pathlib import Path
from typing import Union, Optional, Iterable

from backend_algorithms.export_model.export_utils.export_data import ExportData, ExportImageData, Export3DLidarData, \
    Export4DLidarData, ExportAVData, ExportTextData
from backend_algorithms.utils.general import find_stack


class ExportDataset:
    _data_type = ExportData

    def __init__(
            self,
            root_path,
            logger,
            start_time,
            add_info
    ):
        self._jsons = Path(root_path).rglob('**/data/*.json') if isinstance(root_path, (str, Path)) else iter(root_path)

        self._logger = logger
        self._cnt = 0
        self._script = find_stack(
            target_parts=["export_model", ("custom", "common")]
        )
        self._start_time = start_time if start_time is not None else time.time()
        self._add_info = add_info

    def __iter__(
            self
    ):
        return self

    def __next__(
            self
    ) -> Union[ExportData, ExportImageData, Export3DLidarData, Export4DLidarData, ExportAVData, ExportTextData]:
        meta_path = Path(next(self._jsons))
        self._cnt += 1

        return self._data_type(
            meta_path=meta_path,
            logger=self._logger,
            no=self._cnt,
            script=self._script,
            start_time=self._start_time,
            add_info=self._add_info
        )

    def __str__(
            self
    ):
        return self.__class__.__name__

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}"


class ImageExportDataset(ExportDataset):
    _data_type = ExportImageData

    def __init__(
            self,
            root_path: Union[str, Path, Iterable],
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = ''
    ):
        super().__init__(
            root_path=root_path,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def __next__(
            self
    ) -> ExportImageData:
        return super().__next__()


class Lidar3DExportDataset(ExportDataset):
    _data_type = Export3DLidarData

    def __init__(
            self,
            root_path: Union[str, Path, Iterable],
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = ''
    ):
        super().__init__(
            root_path=root_path,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def __next__(
            self
    ) -> Export3DLidarData:
        return super().__next__()


class Lidar4DExportDataset(ExportDataset):
    _data_type = Export4DLidarData

    def __init__(
            self,
            root_path: Union[str, Path, Iterable],
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = ''
    ):
        super().__init__(
            root_path=root_path,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def __next__(
            self
    ) -> Union[Export3DLidarData, Export4DLidarData]:
        return super().__next__()


class AVExportDataset(ExportDataset):
    _data_type = ExportAVData

    def __init__(
            self,
            root_path: Union[str, Path, Iterable],
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = ''
    ):
        super().__init__(
            root_path=root_path,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def __next__(
            self
    ) -> ExportAVData:
        return super().__next__()


class TextExportDataset(ExportDataset):
    _data_type = ExportTextData

    def __init__(
            self,
            root_path: Union[str, Path, Iterable],
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = ''
    ):
        super().__init__(
            root_path=root_path,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def __next__(
            self
    ) -> ExportTextData:
        return super().__next__()
