import sys
import importlib
import json
from typing import Tuple, List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from backend_algorithms.service.base_post import handle_post
from backend_algorithms.service.post_body import QABody, AddInfo, ImportBody, ExportBody, ImageModelBody
from backend_algorithms.calculate_info.point_cloud_info import cal_point_cloud_info
from backend_algorithms.calculate_info.image_info import cal_image_info

app = FastAPI(title='backend_algorithms')

logger.remove()
logger.add(
    sink=sys.stdout,
    format="<level>{level}</level> | <cyan>{message}</cyan>",
    backtrace=True,
    catch=True,
    enqueue=False
)


@app.post('/pointCloud/additional/information')
def calculate_point_cloud_info(
        data_body: AddInfo
):
    logger.info(f"add point cloud info task start")

    code, message, status_code, result = handle_post(
        cal_point_cloud_info,
        data_body.dict()
    )

    return JSONResponse(
        content={"code": code, "message": message, "data": result},
        status_code=status_code
    )


@app.post('/image/additional/information')
def calculate_image_info(
        data_body: AddInfo
):
    logger.info(f"add image info task start")

    code, message, status_code, result = handle_post(
        cal_image_info,
        data_body.dict()
    )

    return JSONResponse(
        content={"code": code, "message": message, "data": result},
        status_code=status_code
    )


@app.post('/customQaRule/{ruleCode}')
def custom_qa_rule(
        ruleCode: str,
        qa_body: QABody
):
    def _qa(
            rule_code,
            body
    ):
        module = importlib.import_module(
            f"backend_algorithms.qa_rule.{rule_code.replace('-', '.')}"
        )

        return module.detect(json.load(body.filePath.open(encoding="utf-8")))

    _, _, status_code, result = handle_post(
        _qa,
        ruleCode,
        qa_body
    )

    return JSONResponse(
        content=result,
        status_code=status_code
    )


@app.post('/customFormatConversion/import/{formatCode}')
def data_import(
        formatCode: str,
        format_body: ImportBody
):
    def _data_import(
            format_code,
            import_body
    ):
        module = importlib.import_module(
            f"backend_algorithms.import_model.{format_code.replace('-', '.')}"
        )

        return module.trans(import_body.dict())

    logger.info(f"data import task start: {formatCode}")

    code, message, status_code, result = handle_post(
        _data_import,
        formatCode,
        format_body
    )

    if isinstance(result, (Tuple, List)) and len(result) == 2:
        code, message = result

    return JSONResponse(
        content={"code": code, "message": message},
        status_code=status_code
    )


@app.post('/customFormatConversion/export/{formatCode}')
def data_export(
        formatCode: str,
        format_body: ExportBody
):
    def _data_export(
            format_code,
            export_body
    ):
        module = importlib.import_module(
            f"backend_algorithms.export_model.{format_code.replace('-', '.')}"
        )

        return module.trans(export_body.dict())

    logger.info(f"data export task start: {formatCode}")

    code, message, status_code, result = handle_post(
        _data_export,
        formatCode,
        format_body
    )

    if isinstance(result, (Tuple, List)) and len(result) == 2:
        code, message = result

    return JSONResponse(
        content={"code": code, "message": message},
        status_code=status_code
    )


@app.post('/model/image/{modelCode}')
def img_model(
        modelCode: str,
        model_body: ImageModelBody
):
    def _img_model(
            model_code,
            body
    ):
        return importlib.import_module(
            f"backend_algorithms.model.image.{model_code}"
        ).model_run(body.dict())

    code, message, status_code, result = handle_post(
        _img_model,
        modelCode,
        model_body
    )

    return JSONResponse(
        content={"code": code, "message": message, "data": result},
        status_code=status_code
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--access_log', type=bool, default=True)
    parser.add_argument('--log_level', type=str, default="info", nargs='?', help="logging level")
    args = parser.parse_args()

    uvicorn.run(
        app='backend_algorithms.service.app:app',
        host=args.host,
        port=args.port,
        access_log=args.access_log,
        log_level=args.log_level,
        reload=True
    )
