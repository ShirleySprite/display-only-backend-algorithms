import time
from pathlib import Path
from urllib.parse import urlparse, unquote
from itertools import chain

import requests
import pandas as pd
from loguru import logger


def parse_path_from_url(
        url
):
    return Path(unquote(urlparse(url).path).strip('/\\'))


def trans(input_json):
    origin_path = Path(input_json.get('originPath'))
    target_path = Path(input_json.get('targetPath'))

    start_time = time.time()
    no = 1
    for x in chain(origin_path.rglob("*.csv"), origin_path.rglob("*.xlsx")):
        table_path = x.relative_to(origin_path).with_suffix('')
        if x.suffix == ".csv":
            func = pd.read_csv
        else:
            func = pd.read_excel

        url_df = func(x, header=None)
        for url in url_df[0]:
            url_path = parse_path_from_url(url)

            logger.info(
                f'time(s): {time.time() - start_time:.3f}; '
                f'script: upload_by_links; '
                f'no: {no}; '
                f'file: {table_path / url_path}'
            )

            img_output = target_path / table_path / url_path
            img_output.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(url)
            if resp.status_code == 200:
                img_output.write_bytes(resp.content)
            else:
                raise ValueError("download failed")

            no += 1
