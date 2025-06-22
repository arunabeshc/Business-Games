import logging
import sys
from io import StringIO

class LogFormatterAgent:

    log_stream = StringIO()
    html_log_stream = StringIO()
    
    def __init__(self):
        self.logger = logging.getLogger("LogFormatterAgent")
        self.logger.setLevel(logging.INFO)

    class HTMLLogFormatter(logging.Formatter):
        AGENT_COLORS = {
            "FrontierAgent": "blue",
            "LogFormatterAgent": "darkorange",
            "OpenSourceAgent": "#d97706",
            "RAGAgent": "#6b21a8",  # deep purple
            "Default": "black"
        }

        AGENT_BACKGROUNDS = {
            "FrontierAgent": "#eef5ff",
            "LogFormatterAgent": "#fff8e1",
            "OpenSourceAgent": "#fff7ed",
            "RAGAgent": "#f3e8ff",  # soft lilac
            "Default": "#f4f4f4"
        }

        def format(self, record):
            record.asctime = self.formatTime(record, self.datefmt)
            logger_name = record.name.split('.')[-1]

            color = self.AGENT_COLORS.get(logger_name, self.AGENT_COLORS["Default"])
            background = self.AGENT_BACKGROUNDS.get(logger_name, self.AGENT_BACKGROUNDS["Default"])

            return (
                f'<div style="'
                f'background-color:{background}; '
                f'color:{color}; '
                f'padding:6px; '
                f'margin-bottom:2px; '
                f'font-family:monospace; '
                f'white-space:pre-wrap;">'
                f'[{record.asctime}] [{logger_name}] [{record.levelname}] {record.getMessage()}'
                f'</div>'
            )
        
    def init_logging(self):
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Memory buffer handler
        stream_handler = logging.StreamHandler(self.log_stream)
        stream_handler.setLevel(logging.INFO)

        # Formatter for plain logs
        formatter = logging.Formatter(
            "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %z",
        )
        handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # HTML handler for Gradio
        html_stream_handler = logging.StreamHandler(self.html_log_stream)
        html_stream_handler.setLevel(logging.INFO)
        html_stream_handler.setFormatter(self.HTMLLogFormatter(
            fmt="%(asctime)s [Agents] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %z"
        ))

        # Add handlers after they're defined
        root.addHandler(handler)
        root.addHandler(stream_handler)
        root.addHandler(html_stream_handler)

        # This will now be properly captured by all handlers
        self.logger.info("Logging Agent is ready")
