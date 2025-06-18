import logging
from io import StringIO

log_stream = StringIO()
html_log_stream = StringIO()

class HTMLLogFormatter(logging.Formatter):
    AGENT_COLORS = {
        "FrontierAgent": "blue",
        "Default": "black"
    }

    def format(self, record):
        record.asctime = self.formatTime(record, self.datefmt)
        logger_name = record.name.split('.')[-1]  # often "__main__" or module
        color = self.AGENT_COLORS.get(logger_name, self.AGENT_COLORS["Default"])
        return f'<div style="color:{color}">[{record.asctime}] [{logger_name}] [{record.levelname}] {record.getMessage()}</div>'
    
def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Memory buffer handler
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

     # HTML handler for Gradio
    html_stream_handler = logging.StreamHandler(html_log_stream)
    html_stream_handler.setLevel(logging.INFO)
    html_stream_handler.setFormatter(HTMLLogFormatter(
        fmt="%(asctime)s [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z"
    ))
    root.addHandler(html_stream_handler)

    root.addHandler(handler)
    root.addHandler(stream_handler)