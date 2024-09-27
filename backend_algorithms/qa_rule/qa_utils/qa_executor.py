import time
from typing import Union, Optional, List, Dict, Iterator, Generator, Callable, Set

from backend_algorithms.utils.general import AlwaysTrueList
from backend_algorithms.utils.ontology import Ontology
from backend_algorithms.utils.annotation import AnnotationObject
from backend_algorithms.qa_rule.qa_utils.rule_info import ImageRuleInfo, LidarRuleInfo, AVRuleInfo, TextRuleInfo
from backend_algorithms.qa_rule.qa_utils.internal_result import InternalResult, ImageInternalResult, \
    LidarInternalResult, AVInternalResult, TextInternalResult
from backend_algorithms.qa_rule.qa_utils.comment import ImageComment, LidarComment, AVComment, TextComment
from backend_algorithms.utils.general import find_stack


class QAExecutor:
    def __init__(
            self,
            input_json,
            source_type
    ):
        self.ontology_info: Ontology = Ontology(**input_json.get('ontologyInfo') or {})

        if isinstance(source_type, str):
            source_type = [source_type]
        elif source_type is None:
            source_type = AlwaysTrueList()
        self._source_type = source_type

        self.results = []
        self.data_ids: List = input_json.get('dataIds') or []

        self.error_data_ids: Set = set()
        self.error_objects: List = []
        self.data_result_violate_messages = []

        self._cnt = 0
        self._script = find_stack(
            target_parts=["qa_rule", ("custom", "common")]
        )
        self._start_time = time.time()

    @property
    def resp(
            self
    ) -> Dict:
        return {
            "dataIds": list(self.error_data_ids),
            "classifications": [],
            "objects": self.error_objects,
            "dataResultViolateMessages": self.data_result_violate_messages
        }

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(" \
               f"Data: {len(self.data_ids)}" \
               f")"

    def _gen_results(
            self,
            input_json,
            source_type,
            result_cls
    ):
        for x in (input_json.get('data') or []):
            for y in (x.get('annotationResults') or []):
                if y.get('sourceType') in source_type:
                    self._cnt += 1
                    yield result_cls(
                        data=y,
                        no=self._cnt,
                        script=self._script,
                        start_time=self._start_time
                    )

    def update_error_objects(
            self,
            data_id: int,
            instance: AnnotationObject,
            violate_message: Optional[Union[str, Callable]] = None
    ):
        if isinstance(violate_message, str):
            vio = violate_message
        elif isinstance(violate_message, Callable):
            vio = violate_message(instance)
        else:
            vio = ''

        self.error_data_ids.add(data_id)
        self.error_objects.append(
            {'objectId': instance.id, "violateMessage": vio}
        )

    def update_error_objects_totally(
            self,
            trigger: Callable,
            violate_message: Optional[Union[str, Callable]] = None,
            **kwargs
    ):
        for r in self.results:
            for inst in r.instances:
                if trigger(inst, **kwargs):
                    self.update_error_objects(
                        data_id=r.data_id,
                        instance=inst,
                        violate_message=violate_message
                    )

    def update_error_result(
            self,
            result: InternalResult
    ) -> None:
        self.error_data_ids.add(result.data_id)

    def update_error_result_totally(
            self,
            trigger: Callable = lambda x: True,
            **kwargs
    ) -> None:
        for result in self.results:
            if trigger(result, **kwargs):
                self.update_error_result(
                    result=result
                )

    def update_error_result_with_message(
            self,
            result: InternalResult,
            violate_message: Optional[Union[str, Callable]] = None
    ) -> None:
        if isinstance(violate_message, str):
            vio = violate_message
        elif isinstance(violate_message, Callable):
            vio = violate_message(result)
        else:
            vio = ''

        if result.data_id not in self.error_data_ids:
            self.error_data_ids.add(result.data_id)
            self.data_result_violate_messages.append(
                {
                    "dataId": result.data_id,
                    "violateMessage": vio
                }
            )

    def update_error_result_with_message_totally(
            self,
            trigger: Callable = lambda x: True,
            violate_message: Optional[Union[str, Callable]] = None,
            **kwargs
    ) -> None:
        for result in self.results:
            if trigger(result, **kwargs):
                self.update_error_result_with_message(
                    result=result,
                    violate_message=violate_message
                )

    def update_error_attributes(
            self,
            data_id: int,
            instance: AnnotationObject,
            error_attrs: List
    ):
        if error_attrs:
            self.error_data_ids.add(data_id)
            self.error_objects.append(
                {
                    "objectId": instance.id,
                    "attributeIds": error_attrs
                }
            )

    def update_error_attributes_totally(
            self,
            trigger,
            **kwargs
    ):
        for r in self.results:
            for inst in r.instances:
                error_attrs = trigger(inst, **kwargs)
                if error_attrs:
                    self.update_error_attributes(
                        data_id=r.data_id,
                        instance=inst,
                        error_attrs=error_attrs
                    )


class ImageQAExecutor(QAExecutor):
    def __init__(
            self,
            input_json: Dict,
            source_type: Optional[Union[str, List]] = None
    ):
        super().__init__(
            input_json=input_json,
            source_type=source_type
        )

        self.results: Generator[ImageInternalResult, None, None] = self._gen_results(
            input_json=input_json,
            source_type=self._source_type,
            result_cls=ImageInternalResult
        )
        self.scene_id: Optional[int] = input_json.get('sceneId')
        self.rule_info: ImageRuleInfo = ImageRuleInfo(input_json.get('ruleInfo') or {})
        self.comments: Iterator[ImageComment] = map(lambda x: ImageComment(x), input_json.get("comments") or [])


class LidarQAExecutor(QAExecutor):
    def __init__(
            self,
            input_json: Dict,
            source_type: Optional[Union[str, List]] = None
    ):
        super().__init__(
            input_json=input_json,
            source_type=source_type
        )

        self.results: Generator[LidarInternalResult, None, None] = self._gen_results(
            input_json=input_json,
            source_type=self._source_type,
            result_cls=LidarInternalResult
        )
        self.scene_id: Optional[int] = input_json.get('sceneId')
        self.rule_info: LidarRuleInfo = LidarRuleInfo(input_json.get('ruleInfo') or {})
        self.comments: Iterator[LidarComment] = map(lambda x: LidarComment(x), input_json.get("comments") or [])


class AVQAExecutor(QAExecutor):
    def __init__(
            self,
            input_json: Dict,
            source_type: Optional[Union[str, List]] = None
    ):
        super().__init__(
            input_json=input_json,
            source_type=source_type
        )

        self.results: Generator[AVInternalResult, None, None] = self._gen_results(
            input_json=input_json,
            source_type=self._source_type,
            result_cls=AVInternalResult
        )
        self.rule_info: AVRuleInfo = AVRuleInfo(input_json.get('ruleInfo') or {})
        self.comments: Iterator[AVComment] = map(lambda x: AVComment(x), input_json.get("comments") or [])


class TextQAExecutor(QAExecutor):
    def __init__(
            self,
            input_json: Dict,
            source_type: Optional[Union[str, List]] = None
    ):
        super().__init__(
            input_json=input_json,
            source_type=source_type
        )

        self.results: Generator[TextInternalResult, None, None] = self._gen_results(
            input_json=input_json,
            source_type=self._source_type,
            result_cls=TextInternalResult
        )
        self.rule_info: TextRuleInfo = TextRuleInfo(input_json.get('ruleInfo') or {})
        self.comments: Iterator[TextComment] = map(lambda x: TextComment(x), input_json.get("comments") or [])
