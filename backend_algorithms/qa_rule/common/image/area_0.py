from backend_algorithms.qa_rule import ImageQAExecutor


def detect(input_json):
    qa_exe = ImageQAExecutor(
        input_json=input_json
    )
    qa_exe.update_error_objects_totally(
        lambda x: x.type == "POLYGON" and x.area == 0
    )

    return qa_exe.resp
