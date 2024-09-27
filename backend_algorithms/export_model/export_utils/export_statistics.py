import json
from pathlib import Path


class DataStat:
    def __init__(
            self,
            stat
    ):
        self.dict = stat

        self.total = self.dict["total"]
        self.by_status = self.dict["byStatus"]


class ResultStat:
    def __init__(
            self,
            stat
    ):
        self.dict = stat

        self.total = self.dict["total"]
        self.by_objct_type = self.dict["byObjectType"]
        self.by_label = self.dict["byLabel"]


class Statistics:
    def __init__(
            self,
            root_path: Path
    ):
        try:
            self.stat_path = next(root_path.rglob("**/statistic.json"))
            raw_data = json.load(self.stat_path.open(encoding="utf-8"))
            d, r = raw_data["data"], raw_data["result"]
        except StopIteration:
            d, r = {"total": 0, "byStatus": {}}, {"total": 0, "byObjectType": {}, "byLabel": {}}

        self.data: DataStat = DataStat(d)
        self.result: ResultStat = ResultStat(r)

    def to_dict(
            self
    ):
        return {
            "data": self.data.dict,
            "result": self.result.dict
        }
