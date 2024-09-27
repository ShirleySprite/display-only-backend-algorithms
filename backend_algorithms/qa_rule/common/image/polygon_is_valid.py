from backend_algorithms.qa_rule import ImageQAExecutor


def detect(input_json):
    qa_exe = ImageQAExecutor(
        input_json=input_json
    )
    qa_exe.update_error_objects_totally(
        lambda x: x.type == "POLYGON" and not x.to_shape().is_valid
    )

    return qa_exe.resp
