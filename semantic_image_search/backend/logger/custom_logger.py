import os
import sys
import logging
from datetime import datetime
import tempfile
import structlog


class CustomLogger:
    def __init__(self, log_dir="logs"):
        self.log_file_path = None
        is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

        if is_serverless:
            candidate_dir = os.path.join(tempfile.gettempdir(), log_dir)
        else:
            candidate_dir = os.path.join(os.getcwd(), log_dir)

        try:
            os.makedirs(candidate_dir, exist_ok=True)
            log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
            self.log_file_path = os.path.join(candidate_dir, log_file)
        except OSError:
            # Read-only filesystem (e.g. AWS Lambda / Vercel /var/task); stdout logging is used
            self.log_file_path = None

    def get_logger(self, name=__file__):
        logger_name = os.path.basename(name)

        handlers = []
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)

        if self.log_file_path:
            try:
                file_handler = logging.FileHandler(self.log_file_path)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(logging.Formatter("%(message)s"))
                handlers.append(file_handler)
            except OSError:
                pass

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=handlers,
            force=True,
        )

        # Configure structlog for JSON structured logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)