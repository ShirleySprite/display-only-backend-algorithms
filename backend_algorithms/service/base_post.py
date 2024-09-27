import traceback

from loguru import logger


def handle_post(
        func,
        *args,
        **kwargs
):
    result = None

    try:
        result = func(*args, **kwargs)
        code = "OK"
        message = ""
        status_code = 200
    except:
        trace = traceback.format_exc()
        if "ModuleNotFoundError" in trace and trace.count("backend-algorithms") < 3:
            code = "ERROR"
            message = "Wrong URL!"
            status_code = 404
        else:
            logger.error(trace)
            code = "ERROR"
            message = "Python script error!"
            status_code = 500

    return code, message, status_code, result
