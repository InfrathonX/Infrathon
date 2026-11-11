"""No-op compatibility timing hooks; evaluator timing is external."""

from contextlib import ContextDecorator


class TimeLoggingContext(ContextDecorator):
    def __init__(self, operation_type: str, logging_level: int = 0) -> None:
        self.operation_type = operation_type
        self.logging_level = logging_level

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


time_logging_decorator = TimeLoggingContext


def print_operator_log_data(module=None, inputs=None, output=None) -> None:
    return None
