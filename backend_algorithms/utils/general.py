import inspect
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import UserList
from typing import List, Iterable, Callable, Dict, Any, Union, Tuple

from treelib import Tree


def filter_parts(
        org_path: Union[str, Path],
        filters: Union[Tuple, str]
) -> Path:
    if isinstance(filters, str):
        filters = filters,

    return Path(*filter(lambda x: x not in filters, Path(org_path).parts))


def inplace_filter(
        key,
        lst
):
    i = 0
    n = len(lst)
    while i < n:
        if not key(lst[i]):
            lst.pop(i)
            n -= 1
        else:
            i += 1


def hex_to_rgb(
        hex_code
):
    # 去除十六进制代码中的 '#' 符号（如果有）
    hex_code = hex_code.strip('#')

    # 将十六进制代码分割成 R、G、B 三部分
    red = int(hex_code[0:2], 16)
    green = int(hex_code[2:4], 16)
    blue = int(hex_code[4:6], 16)

    return red, green, blue


def drop_duplicates(
        dups: Iterable,
        key: Callable = lambda x: x
) -> List:
    result = {key(x): x for x in dups}

    return list(result.values())


def reg_dict(
        org_dict: Dict,
        keys: List,
        default_value: Any = 0
) -> Dict:
    result = {k: default_value for k in keys}
    result.update(org_dict)

    return result


def groupby(
        items: Iterable,
        func: Callable,
        assign_keys=None
) -> Dict:
    if assign_keys:
        result = {k: [] for k in assign_keys}
    else:
        result = {}

    for x in items:
        k = func(x)
        result.setdefault(k, []).append(x)

    return result


def onto_id_map(
        old_onto,
        new_onto
):
    def _update_map(
            old_part,
            new_part
    ):
        if type(new_part) is not type(old_part):
            return

        if isinstance(old_part, Dict):
            if "id" in old_part:
                id_map[old_part["id"]] = new_part.get("id")
            for k, v in old_part.items():
                _update_map(v, new_part.get(k))

        elif isinstance(old_part, List):
            for o, n in zip(old_part, new_part):
                _update_map(o, n)

    id_map = {}
    _update_map(old_onto, new_onto)

    return id_map


def change_onto_id(
        data_path,
        id_map
):
    for x in data_path.rglob("**/result/*.json"):
        result = x.read_text()
        for k, v in id_map.items():
            result = result.replace(str(k), str(v))
        x.write_text(result)


# 字符串
def del_redundant_spaces(text):
    while "  " in text:
        text = text.replace("  ", ' ')

    return text


# 时间

def get_beijing_time():
    beijing_timezone = timezone(timedelta(hours=8))
    current_time_beijing = datetime.now(beijing_timezone)

    return current_time_beijing.strftime("%Y-%m-%dT%H:%M:%S%Z")


def get_current_time():
    return datetime.now().strftime('%y-%m-%d %H:%M:%S')


def seconds_to_hms(
        seconds
):
    from decimal import Decimal

    seconds = Decimal(str(seconds))

    # 计算小时、分钟和秒
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = round((seconds - int(seconds)) * 1000)

    # 将结果格式化为字符串
    time_string = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    return time_string


def sliding_window(
        lst: List,
        n
) -> List:
    lst_length = len(lst)
    if n > lst_length:
        raise ValueError("Window size n should not be larger than the length of the list.")

    for i in range(lst_length - n + 1):
        yield lst[i:i + n]


def compare_files(
        source_path,
        target_path
):
    source_path = Path(source_path)
    target_path = Path(target_path)

    for file1 in source_path.rglob('*'):
        if not file1.is_file():
            continue

        file2 = target_path / file1.relative_to(source_path)

        yield file1.read_bytes() == file2.read_bytes()


# 自定义数据类型

class AlwaysTrueList(UserList):
    def __contains__(
            self,
            item
    ):
        return True


class CustomTree(Tree):
    def to_dict(self, nid=None, key=None, sort=True, reverse=False, with_data=False) -> Dict:
        """Transform the whole tree into a dict."""

        nid = self.root if (nid is None) else nid
        ntag = self[nid].tag

        tree_dict = {ntag: {}}
        queue = [self[i] for i in self[nid].successors(self._identifier)]
        key = (lambda x: x) if (key is None) else key
        if sort:
            queue.sort(key=key, reverse=reverse)

        for elem in queue:
            tree_dict[ntag].update(
                self.to_dict(
                    elem.identifier, with_data=with_data, sort=sort, reverse=reverse
                )
            )
        if len(tree_dict[ntag]) == 0:
            tree_dict = {ntag: None} if not with_data else {ntag: self[nid].data}

        return tree_dict


def find_stack(
        target_parts: List[Union[str, Tuple]],
        n_parts: int = 4
) -> str:
    for s in inspect.stack():
        cur_file = Path(s.filename)
        pts = cur_file.parts
        pts_set = set(pts)

        is_match = True
        for p in target_parts:
            if isinstance(p, str):
                if p not in pts:
                    is_match = False
            else:
                if not pts_set.intersection(set(p)):
                    is_match = False

        if is_match:
            return '/'.join(pts[-n_parts:])

    return 'unknwon'
