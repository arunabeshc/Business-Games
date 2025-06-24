# imports
import gradio as gr
import modal
from agents.frontier_agent import FrontierAgent
from agents.open_source_agent import OpenSourceAgent
from agents.RAG_agent import RAGAgent
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import logging
import sys
from io import StringIO

# Step 1: Create Modal image
image = (
    modal.Image.debian_slim()
    .pip_install([
        "gradio", "openai", "chromadb", "python-dotenv",
        "scikit-learn", "sentence_transformers", "transformers", "torch"
    ])
    .workdir("/app")
    .add_local_dir("agents", "/app/agents")
    .add_local_dir("agile_process", "/app/agile_process")
    .add_local_file(".env", "/app/.env")
    .add_local_file("struct_my_proj.py", "/app/struct_my_proj.py")
    .add_local_dir("classes", "/app/classes")
    .add_local_dir("models", "/app/models")
)

# Step 2: Define your Modal App
app = modal.App("business-chatbot")

@app.function(
    image=image,
    timeout=600,
    cpu=4,
    memory=8192,
    secrets=[
        modal.Secret.from_name("all-secrets"),
        modal.Secret.from_name("hf-secret")
    ]
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
# Initialization
def ui():

    log_stream = StringIO()
    html_log_stream = StringIO()

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
        
    def init_logging():
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Memory buffer handler
        stream_handler = logging.StreamHandler(log_stream)
        stream_handler.setLevel(logging.INFO)

        # Formatter for plain logs
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

        # Add handlers after they're defined
        root.addHandler(handler)
        root.addHandler(stream_handler)
        root.addHandler(html_stream_handler)

    init_logging()

    ra=RAGAgent()
    fa=FrontierAgent(ra)
    oa=OpenSourceAgent(ra)

    def get_html_logs():
        html_log_stream.seek(0)
        return html_log_stream.read()

    def clear_logs():
        log_stream.truncate(0)
        log_stream.seek(0)
        html_log_stream.truncate(0)
        html_log_stream.seek(0)
        return None, ""

    def chat(history,Model):
        if Model=="Open AI (gpt-4o-mini)":
            history = fa.chat_open_ai(history)
        elif Model=="Open Source (HuggingFace Llama-3.1)":
            history = oa.chat_llama(history)
        return history
        
    initial_prompt = """👋 Hello! How can I assist you today? If you're looking to set up a new data engineering project, please provide me with details about the source systems, the number of target data marts, and a brief outline of the overall architecture."""

    # with gr.Blocks(css="""
    #     #log_box {
    #         height: 300px;
    #         overflow-y: scroll;
    #         border: 1px solid #ccc;
    #         padding: 10px;
    #         background: #f9f9f9;
    #         font-family: monospace;
    #         white-space: pre-wrap;
    #     }
    # """) as ui:
    #     with gr.Row():
    #         with gr.Column(scale=1):
    #             chatbot = gr.Chatbot(height=300, type="messages")
    #         with gr.Column(scale=1):
                
    #             logs_box = gr.HTML(label="Logs", elem_id="log_box")
    #     with gr.Row():
    #         Model = gr.Dropdown(["Open AI (gpt-4o-mini)","Open Source (HuggingFace Llama-3.1)"],
    #                             multiselect=False,
    #                             label="Model",
    #                             interactive=True)
    #     with gr.Row():
    #         entry = gr.Textbox(label="Chat with our AI Assistant:")
    #     with gr.Row():
    #         clear = gr.Button("Clear")


    #     # timer = gr.Timer(value=2, active=True)
    #     # timer.tick(get_html_logs, inputs=None, outputs=[logs_box])
    #     refresh = gr.Button("🔄 Refresh Logs")
    #     refresh.click(get_html_logs, outputs=[logs_box])

        # def do_entry(message, history):
        #     history += [{"role":"user", "content":message}]
        #     logging.info(f"User message: {message}")
        #     yield "", history, get_html_logs()
            
    #     def set_initial_prompt():
    #         return [{"role": "assistant", "content": initial_prompt}]
        
    #     # ui.load(set_initial_prompt, inputs=None, outputs=chatbot)
        
    #     entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot, logs_box]).then(
    #         chat, inputs=[chatbot, Model], outputs=[chatbot]
    #     )
        
    #     clear.click(clear_logs, inputs=None, outputs=[chatbot, logs_box], queue=False)

    with gr.Blocks() as ui:
        chatbot = gr.Chatbot()
        logs_box = gr.HTML()
        def do_entry(message, history):
            history += [{"role":"user", "content":message}]
            logging.info(f"User message: {message}")
            yield "", history, get_html_logs()
        Model = gr.Dropdown(["Open AI (gpt-4o-mini)","Open Source (HuggingFace Llama-3.1)"],
                                multiselect=False,
                                label="Model",
                                interactive=True)
        entry = gr.Textbox()
        clear = gr.Button("Clear")
        refresh = gr.Button("🔄 Refresh Logs")
        refresh.click(get_html_logs, outputs=[logs_box])
        entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot, logs_box]).then(
            chat, inputs=[chatbot, Model], outputs=[chatbot]
        )
        clear.click(clear_logs, inputs=None, outputs=[chatbot, logs_box])

    fastapi_app = FastAPI()

    @fastapi_app.get("/manifest.json", include_in_schema=False)
    async def stub_manifest():
        return {}

    ui.queue(max_size=5)
    # Mount Gradio UI onto the FastAPI app
    return mount_gradio_app(app=fastapi_app, blocks=ui, path="/")