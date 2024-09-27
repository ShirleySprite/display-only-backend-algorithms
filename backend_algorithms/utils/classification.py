from typing import Dict, List

from backend_algorithms.utils.general import CustomTree


class Classification:
    def __init__(
            self,
            classification: Dict
    ):
        self.raw: Dict = classification
        self.classification_id: int = int(self.raw['classificationId'])
        self.values: List = self.raw.get('values') or []

    def __repr__(
            self
    ):
        return f"{self.__class__.__name__}(classification_id: {self.classification_id})"

    @property
    def tree(
            self
    ) -> Dict:
        tree = CustomTree()
        tree.create_node("Root", "root")
        for v in self.values:
            tree.create_node(
                tag=v["name"],
                identifier=v["id"],
                parent=v.get("pid") or "root",
                data=v["value"]
            )

        return tree.to_dict(with_data=True)["Root"]

    @property
    def simp(
            self
    ) -> Dict:
        return {v['name']: v['value'] for v in self.values}
