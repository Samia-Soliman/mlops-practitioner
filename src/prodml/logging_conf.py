import logging
from contextvars import ContextVar

from pythonjsonlogger import json

correlation_id_var: ContextVar[str] = ContextVar(
    "correlation_id",
    default="_",
)


def configure_logging() -> None:
    """
    Configures logging to output JSON formatted logs.
    """
    logger = logging.getLogger()

    logger.handlers.clear()
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.correlation_id = correlation_id_var.get()
        return record

    logging.setLogRecordFactory(record_factory)

    log_handler = logging.FileHandler("app.log")

    formatter = json.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s"
    )

    log_handler.setFormatter(formatter)

    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)
