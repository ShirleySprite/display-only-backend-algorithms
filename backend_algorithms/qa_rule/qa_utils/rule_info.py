import math
from typing import Dict, List, Optional, Union

from backend_algorithms.utils.general import AlwaysTrueList


class RuleInfo:
    def __init__(
            self,
            rule_info: Dict
    ):
        self.raw: Dict = rule_info
        self.id: Optional[int] = self.raw.get('id')
        self._targeted_config = self.raw.get('targetedConfig') or {}
        self._filter = self.raw.get('fillRuleParameter') or {}
        self.filter_classes: Union[List, AlwaysTrueList] = self._filter.get('classes') or AlwaysTrueList()

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}"


class ImageRuleInfo(RuleInfo):
    def __init__(
            self,
            rule_info: Dict
    ):
        super().__init__(
            rule_info=rule_info
        )


class LidarRuleInfo(RuleInfo):
    def __init__(
            self,
            rule_info: Dict
    ):
        super().__init__(
            rule_info=rule_info
        )

        r = self._targeted_config.get('radius') or {}
        self.rmin: Union[float, int] = r.get('min') or 0
        self.rmax: Union[float, int] = r.get('max') or math.inf

        h = self._targeted_config.get('height') or {}
        self.hmin: Union[float, int] = h.get('min') or -math.inf
        self.hmax: Union[float, int] = h.get('max') or math.inf


class AVRuleInfo(RuleInfo):
    def __init__(
            self,
            rule_info: Dict
    ):
        super().__init__(
            rule_info=rule_info
        )


class TextRuleInfo(RuleInfo):
    def __init__(
            self,
            rule_info: Dict
    ):
        super().__init__(
            rule_info=rule_info
        )
