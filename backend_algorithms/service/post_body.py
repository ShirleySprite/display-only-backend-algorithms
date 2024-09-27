from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel


class QABody(
    BaseModel
):
    filePath: Path


class AddInfo(BaseModel):
    instances: List
    segments: List


class ExportBody(BaseModel):
    hasOriginFile: Optional[bool]
    originPath: str
    targetPath: str
    datasetClassList: List
    datasetClassificationList: List


class ImportBody(BaseModel):
    originPath: str
    targetPath: str
    datasetClassList: List
    datasetClassificationList: List


class ImageModelBody(BaseModel):
    class DataBody(BaseModel):
        id: int
        url: str

    data: DataBody
