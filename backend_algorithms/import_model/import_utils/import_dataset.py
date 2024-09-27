import time
from pathlib import Path
from itertools import chain
from typing import Union, Tuple, Callable, Hashable, Optional

from backend_algorithms.import_model.import_utils.import_data import ImportImageData, ImportLidarData, \
    ImportAVData
from backend_algorithms.utils.general import find_stack, groupby


class ImportDataset:
    def __init__(
            self,
            root_path,
            pattern,
            data_type,
            logger,
            start_time,
            add_info
    ):
        self._root_path = Path(root_path)
        self._pattern = [pattern] if isinstance(pattern, str) else pattern
        self._data_type = data_type
        self._files = chain(*[self._root_path.rglob(x) for x in self._pattern])

        self._logger = logger
        self._cnt = 0
        self._script = find_stack(
            target_parts=["import_model", ("custom", "common")]
        )
        self._start_time = start_time if start_time is not None else time.time()
        self._add_info = add_info

    def __iter__(
            self
    ):
        return self

    def __next__(
            self
    ):
        meta_path = next(self._files)
        while "__MACOSX" in meta_path.parts:
            meta_path = next(self._files)
        self._cnt += 1

        return self._data_type(
            root_path=self._root_path,
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
        return f"{self.__class__.__name__}(root_path=r'{self._root_path}')"


class ImportImageDataset(ImportDataset):
    def __init__(
            self,
            root_path: Union[str, Path],
            pattern: Union[Tuple, str] = ("**/*.jpg", "**/*.png"),
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = ''
    ):
        super().__init__(
            root_path=root_path,
            pattern=pattern,
            data_type=ImportImageData,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def __next__(
            self
    ) -> ImportImageData:
        return super().__next__()


class ImportLidarDataset(ImportDataset):
    def __init__(
            self,
            root_path: Union[str, Path],
            pattern: Union[Tuple, str] = "**/*.pcd",
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = ''
    ):
        super().__init__(
            root_path=root_path,
            pattern=pattern,
            data_type=ImportLidarData,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def __next__(
            self
    ) -> ImportLidarData:
        return super().__next__()

    def gen_img_map(
            self,
            match_identifier: Callable[[Path], Hashable] = lambda x: x.stem
    ):
        imgs = []
        for file_path in self._root_path.rglob("*.*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() in [".jpeg", ".jpg", ".png", ".tiff"]:
                imgs.append(file_path)

        return groupby(imgs, match_identifier)


class ImportAVDataset(ImportDataset):
    def __init__(
            self,
            root_path: Union[str, Path],
            pattern: Union[Tuple, str],
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = ''
    ):
        super().__init__(
            root_path=root_path,
            pattern=pattern,
            data_type=ImportAVData,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def __next__(
            self
    ) -> ImportAVData:
        return super().__next__()
