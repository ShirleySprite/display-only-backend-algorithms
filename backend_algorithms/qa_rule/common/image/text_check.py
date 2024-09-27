import re

from backend_algorithms.qa_rule import ImageQAExecutor


def _check(
        inst
):
    results = []
    for cv in inst.class_values:
        if cv.get('type') == 'TEXT':
            text = cv.get("value") or ''
            results.append(
                text != text.strip() or
                bool(re.search(r'[\n\t]', text))
            )

    return any(results)


def detect(input_json):
    qa_exe = ImageQAExecutor(
        input_json=input_json
    )
    qa_exe.update_error_objects_totally(_check)

    return qa_exe.resp
