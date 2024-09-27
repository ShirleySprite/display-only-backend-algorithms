import json
from pathlib import Path
from typing import Callable, Optional, Union, Dict, List

from backend_algorithms.utils.ontology import Ontology
from backend_algorithms.utils.general import groupby
from backend_algorithms.export_model.export_utils.export_data import ExportData
from backend_algorithms.export_model.export_utils.export_dataset import ExportDataset, ImageExportDataset, \
    Lidar3DExportDataset, Lidar4DExportDataset, AVExportDataset, TextExportDataset
from backend_algorithms.export_model.export_utils.export_statistics import Statistics


class ExportExecutor:
    _dataset_type = ExportDataset

    def __init__(
            self,
            input_json
    ):
        self.has_origin_file = input_json.get('hasOriginFile')
        self.ontology_info = Ontology(
            classes=input_json.get("datasetClassList"),
            classifications=input_json.get("datasetClassificationList")
        )
        self.origin_path = Path(input_json.get('originPath'))
        self.target_path = Path(input_json.get('targetPath'))

        self.statistics = Statistics(self.origin_path)

    def iter_dataset(
            self,
            logger: bool = True,
            start_time: Optional[float] = None,
            add_info: str = '',
            files: Optional[List] = None,
    ):
        if files:
            root_path = files
        else:
            root_path = self.origin_path

        return self._dataset_type(
            root_path=root_path,
            logger=logger,
            start_time=start_time,
            add_info=add_info
        )

    def export_statistics(
            self,
            data: Optional[Dict] = None
    ):
        if data is None:
            data = self.statistics.to_dict()

        self.target_path.mkdir(parents=True, exist_ok=True)

        json.dump(
            data,
            (self.target_path / self.statistics.stat_path.name).open('w', encoding="utf-8"),
            indent='    ',
            ensure_ascii=False
        )

    def convert_single_data(
            self,
            single_data: Union[ExportData],
            convert_func: Callable,
            result_ext: str = ".json",
            write_params: Optional[Dict] = None,
            **kwargs
    ):
        result = convert_func(single_data, **kwargs)
        single_data.write_result(
            result=result,
            target_path=self.target_path,
            file_ext=result_ext,
            add_params=write_params
        )

        return result

    def convert_all_data(
            self,
            convert_func,
            result_ext: str = ".json",
            files: Optional[List] = None,
            write_params: Optional[Dict] = None,
            **kwargs
    ):
        for single_data in self.iter_dataset(files=files):
            self.convert_single_data(
                single_data=single_data,
                convert_func=convert_func,
                result_ext=result_ext,
                write_params=write_params,
                **kwargs
            )

    def groupby_dataset(
            self,
            groupby_key,
            sort_key: Optional[Callable] = None
    ):
        def _gen_data(
                data_list,
                start_no
        ):
            for i, d in enumerate(data_list):
                d.log_self(
                    no=start_no + i
                )
                yield d

        dataset = self.iter_dataset(logger=False)
        if sort_key is not None:
            dataset = sorted(dataset, key=sort_key)

        groupby_dataset = {}
        start = 1
        for k, v in groupby(
                items=dataset,
                func=groupby_key
        ).items():
            groupby_dataset[k] = _gen_data(
                data_list=v,
                start_no=start
            )
            start += len(v) + 1

        return groupby_dataset

    def del_invalid_data(
            self
    ):
        for file in self.origin_path.rglob("**/result/*.json"):
            if '"validity":"INVALID",' in file.read_text(encoding="utf-8"):
                data_file = file.parent.parent / "data" / file.name
                data_file.unlink()
                file.unlink()

    def copy_export_files(
            self
    ):
        for x in self.origin_path.rglob("*.*"):
            x_output = self.target_path / x.relative_to(self.origin_path)
            x_output.parent.mkdir(parents=True, exist_ok=True)
            x_output.write_bytes(x.read_bytes())


class ImageExportExecutor(ExportExecutor):
    _dataset_type = ImageExportDataset

    def copy_original_files(
            self
    ):
        if self.has_origin_file:
            for x in self.iter_dataset():
                x.file.replace(
                    x.prepare_path(
                        self.target_path,
                        x.org_path.suffix
                    )
                )


class Lidar3DExportExecutor(ExportExecutor):
    _dataset_type = Lidar3DExportDataset


class Lidar4DExportExecutor(ExportExecutor):
    _dataset_type = Lidar4DExportDataset


class AVExportExecutor(ExportExecutor):
    _dataset_type = AVExportDataset


class TextExportExecutor(ExportExecutor):
    _dataset_type = TextExportDataset
