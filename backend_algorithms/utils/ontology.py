from typing import Dict, List, Optional
from backend_algorithms.utils.general import drop_duplicates


class OntoClf:
    def __init__(
            self,
            classification: Dict
    ):
        self.raw: Dict = classification
        self.id: str = self.raw['id']
        self.name: str = self.raw.get('name') or ''
        self.attribute: Dict = self.raw.get('attribute') or {}
        self.is_required: bool = bool(self.raw.get('isRequired'))

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(id: {self.id}; name: {self.name})"


class OntoCls:
    def __init__(
            self,
            onto_class: Dict
    ):
        self.raw: Dict = onto_class
        self.id: str = self.raw['id']
        self.number: int = self.raw.get("number")
        self.name: str = self.raw.get('name') or ''
        self.alias: str = self.raw.get('alias') or ''
        self.color: Optional[str] = self.raw.get('color')
        self.tool_type: str = self.raw['toolType']
        self.tool_type_options: Dict = self.raw.get('toolTypeOptions') or {}
        self.attributes: List[Dict] = self.raw.get('attributes') or []

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(id: {self.id}; name: {self.name})"


class Ontology:
    def __init__(
            self,
            classes: Optional[List[Dict]] = None,
            classifications: Optional[List[Dict]] = None
    ):
        if classes is None:
            classes = []
        if classifications is None:
            classifications = []

        self.raw: Dict = {
            "classes": classes,
            "classifications": classifications
        }
        self.classifications: List[OntoClf] = [
            OntoClf(x)
            for x in classifications
        ]
        self.classes: List[OntoCls] = [
            OntoCls(x)
            for x in classes
        ]

        self.class_map: Dict = {
            x.id: x
            for x in self.classes
        }

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(" \
               f"Classifications: {len(self.classifications)}; " \
               f"Classes: {len(self.classes)}" \
               f")"

    @property
    def id_name_map(
            self
    ):
        return {x.id: x.name for x in self.classes}

    @property
    def name_id_map(
            self
    ):
        return {x.name: x.id for x in self.classes}

    @property
    def id_alias_map(
            self
    ):
        return {x.id: x.alias for x in self.classes}

    @property
    def id_color_map(
            self
    ):
        return {x.id: x.color for x in self.classes}

    @property
    def id_number_map(
            self
    ):
        return {x.id: x.number for x in self.classes if isinstance(x.number, int)}

    def to_coco_cat(
            self
    ):
        cats = [
            {
                "id": c.number,
                "name": c.name,
                "supercategory": ""
            }
            for c in self.classes
            if isinstance(c.number, int)
        ]
        cats.sort(key=lambda x: x["id"])

        return drop_duplicates(
            cats,
            key=lambda x: x["id"]
        )

    def get_default_map(
            self
    ):
        default_map = {}
        for x in self.classes:
            default_map.setdefault(x.id, {})
            for attr in x.attributes:
                attr_name = attr["name"]
                for op in attr["options"]:
                    if op.get("checked"):
                        default_map[x.id][attr_name] = op["name"]

        return default_map
