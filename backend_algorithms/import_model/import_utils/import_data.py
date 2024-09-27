import time
import json
from pathlib import Path
from typing import Union, Optional, Dict

from loguru import logger


class ImportData:
    def __init__(
            self,
            root_path,
            meta_path,
            **kwargs
    ):
        self.root_path: Path = Path(root_path)
        self.meta_path: Path = Path(meta_path)

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
        return str(self.meta_path)

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(meta_path=r'{self.meta_path}')"

    def log_self(
            self,
            add_info: Optional[str] = None
    ):
        logger.info(
            f'time(s): {time.time() - self._start_time:.3f}; '
            f'script: {self._script}; '
            f'no: {self._no}; '
            f'file: {self.meta_path}; '
            f'add_info: {self._add_info if add_info is None else add_info}'
        )

    def prepare_target_path(
            self,
            target_path: Union[str, Path],
    ):
        output_path = Path(target_path)
        if not output_path.suffix:
            output_path = output_path / self.meta_path.relative_to(self.root_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def to_target_path(
            self,
            target_path: Union[str, Path],
            is_move: bool = False
    ):
        output_path = self.prepare_target_path(target_path)

        if is_move:
            self.meta_path.replace(output_path)
        else:
            output_path.write_bytes(self.meta_path.read_bytes())

        return output_path

    @staticmethod
    def _write_result(
            json_output: Path,
            result: Dict
    ) -> Path:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

        return json_output


class ImportImageData(ImportData):
    def write_result(
            self,
            img_output: Path,
            result: Dict
    ):
        return self._write_result(
            json_output=img_output.parent / "result" / img_output.with_suffix(".json").name,
            result=result
        )


class ImportLidarData(ImportData):
    def prepare_target_path(
            self,
            target_path: Union[str, Path]
    ):
        output_path = target_path / self.meta_path.relative_to(self.root_path)
        output_path = output_path.parent / "lidar_point_cloud_0" / output_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    @staticmethod
    def write_camera_config(
            lid_output: Path,
            config
    ):
        config_output = lid_output.parent.parent / "camera_config" / lid_output.with_suffix(".json").name
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_text(json.dumps(config), encoding="utf-8")

    @staticmethod
    def write_image(
            lid_output: Path,
            img_id: int,
            img_org_path: Path
    ):
        img_output = lid_output.parent.parent / f"camera_image_{img_id}" / lid_output.with_suffix(
            img_org_path.suffix).name
        img_output.parent.mkdir(parents=True, exist_ok=True)
        img_output.write_bytes(img_org_path.read_bytes())

    def write_result(
            self,
            pcd_output: Path,
            result: Dict
    ):
        return self._write_result(
            json_output=pcd_output.parent.parent / "result" / pcd_output.with_suffix(".json").name,
            result=result
        )

    @staticmethod
    def write_ego_config(
            lid_output: Path,
            config
    ):
        config_output = lid_output.parent.parent / "ego_vehicle_config" / lid_output.with_suffix(".json").name
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_text(json.dumps(config), encoding="utf-8")


class ImportAVData(ImportData):
    def write_result(
            self,
            av_output: Path,
            result: Dict
    ):
        return self._write_result(
            json_output=av_output.parent / "result" / av_output.with_suffix(".json").name,
            result=result
        )
