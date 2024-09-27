from typing import Dict, List, Union, Optional, Callable

from backend_algorithms.utils.classification import Classification
from backend_algorithms.utils.annotation import ImageInstance, ImageGroup, LidarInstance, AVInstance, Entity, \
    Relation, ImageSegment, LidarSegment
from backend_algorithms.utils.general import groupby, inplace_filter


class BaseResult:
    def __init__(
            self,
            classifications: List[Dict],
            instance_map: Dict
    ):
        self.classifications: List[Classification] = [Classification(x) for x in classifications]

        self.instances = []
        self._instance_map = instance_map

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(" \
               f"classifications: {len(self.classifications)}; " \
               f"instances: {len(self.instances)}; " \
               f")"

    @property
    def simp_clfs(
            self
    ) -> Dict:
        total = {}
        for x in self.classifications:
            total.update(x.simp)

        return total

    @property
    def tree_clf(
            self
    ) -> Optional[Dict]:
        result = {}
        for clf in self.classifications:
            result.update(clf.tree)

        return result

    def filter_instances(
            self,
            key: Callable,
            inplace: bool = False
    ):
        if inplace:
            inplace_filter(key=key, lst=self.instances)
        else:
            return [x for x in self.instances if key(x)]


class ImageBaseResult(BaseResult):
    def __init__(
            self,
            classifications: List[Dict],
            instances: List[Dict],
            segments: List[Dict],
            instance_map: Dict
    ):
        super().__init__(
            classifications=classifications,
            instance_map=instance_map
        )

        self.instances: List[Union[ImageInstance, ImageGroup]] = [
            self._instance_map[x['type']](x)
            for x in instances
        ]

        self.segments: List[ImageSegment] = [
            self._instance_map["MASK"](x)
            for x in segments
        ]

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(" \
               f"classifications: {len(self.classifications)}; " \
               f"instances: {len(self.instances)}; " \
               f"segments: {len(self.segments)}" \
               f")"

    def get_groups(
            self
    ) -> Dict[ImageGroup, List[ImageInstance]]:
        groups = []
        true_instances = []
        for x in self.instances:
            if x.type == 'GROUP':
                groups.append(x)
            else:
                true_instances.append(x)

        result = {}
        for g in groups:
            result[g] = [x for x in true_instances if g.id in x.groups]

        return result

    def group_all_instances(
            self
    ):
        true_insts = [y for y in self.instances if y.type != "GROUP"]

        return groupby(true_insts, func=lambda x: tuple(x.groups) or (x.id,))


class LidarBaseResult(BaseResult):
    def __init__(
            self,
            classifications: List[Dict],
            instances: List[Dict],
            segments: List[Dict],
            instance_map: Dict
    ):
        super().__init__(
            classifications=classifications,
            instance_map=instance_map
        )

        self.instances: List[LidarInstance] = [
            self._instance_map[x['type']](x)
            for x in instances
        ]

        self.segments: List[LidarSegment] = [
            self._instance_map["SEGMENTATION"](x)
            for x in segments
        ]

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(" \
               f"classifications: {len(self.classifications)}; " \
               f"instances: {len(self.instances)}; " \
               f"segments: {len(self.segments)}" \
               f")"

    def group_23d_instances(
            self,
            track_id: bool = False
    ) -> Dict[str, Dict[str, List[LidarInstance]]]:
        g1 = groupby(self.instances, func=lambda x: x.track_id if track_id else x.track_name)

        return {k: groupby(v, func=lambda x: x.type) for k, v in g1.items()}


class AVBaseResult(BaseResult):
    def __init__(
            self,
            classifications: List[Dict],
            instances: List[Dict],
            instance_map: Dict
    ):
        super().__init__(
            classifications=classifications,
            instance_map=instance_map
        )

        self.instances: List[AVInstance] = [
            self._instance_map[x['type']](x)
            for x in instances
        ]

        self.instances.sort(key=lambda x: x.start)


class TextBaseResult(BaseResult):
    def __init__(
            self,
            classifications: List[Dict],
            entities: List[Dict],
            relations: List[Dict],
            instance_map: Dict
    ):
        super().__init__(
            classifications=classifications,
            instance_map=instance_map
        )

        self.entities: List[Entity] = [
            self._instance_map[x['type']](x)
            for x in entities
        ]
        self.relations: List[Relation] = [
            self._instance_map[x['type']](x)
            for x in relations
        ]

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(" \
               f"classifications: {len(self.classifications)}; " \
               f"entities: {len(self.entities)}; " \
               f"relations: {len(self.relations)}" \
               f")"
