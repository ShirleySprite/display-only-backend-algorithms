from typing import Dict, List, Optional

from backend_algorithms.utils.general import reg_dict


class Comment:
    def __init__(
            self,
            comment: Dict
    ):
        self.raw: Dict = comment
        self.id: str = self.raw['id']
        self.data_id: int = self.raw.get('dataId')
        self.object_id: str = self.raw.get('objectId') or ''
        self.class_id: Optional[int] = self.raw.get('classId')
        self.types: List[str] = self.raw.get('types') or []
        self.content: str = self.raw.get('content') or ''
        self.status: str = self.raw.get('status') or 'OPEN'

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(" \
               f"types: {self.types}; " \
               f"content: {self.content}; " \
               f"status: {self.status}" \
               f")"


class ImageComment(Comment):
    def __init__(
            self,
            comment: Dict
    ):
        super().__init__(
            comment=comment
        )

        self.position: Dict = reg_dict(
            org_dict=self.raw.get('position') or {},
            keys=['x', 'y']
        )


class LidarComment(Comment):
    def __init__(
            self,
            comment: Dict
    ):
        super().__init__(
            comment=comment
        )

        self.position: Dict = reg_dict(
            org_dict=self.raw.get('position') or {},
            keys=['x', 'y', 'z']
        )


class AVComment(Comment):
    def __init__(
            self,
            comment: Dict
    ):
        super().__init__(
            comment=comment
        )


class TextComment(Comment):
    def __init__(
            self,
            comment: Dict
    ):
        super().__init__(
            comment=comment
        )
