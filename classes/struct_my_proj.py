# imports

import gradio as gr
from classes import log_formatter
from agents import frontier_agent

lf=log_formatter()
fa=frontier_agent()

def get_html_logs():
    lf.html_log_stream.seek(0)
    return lf.html_log_stream.read()

def clear_logs():
    lf.log_stream.truncate(0)
    lf.log_stream.seek(0)
    lf.html_log_stream.truncate(0)
    lf.html_log_stream.seek(0)
    return None, ""

lf.init_logging()

def chat(history,Model):
    if Model=="Open AI":
        history = fa.chat_open_ai(history)
    return history
    
initial_prompt = """👋 Hello! How can I assist you today? If you're looking to set up a new data engineering project, please provide me with details about the source systems, the number of target data marts, and a brief outline of the overall architecture."""

with gr.Blocks(css="""
    #log_box {
        height: 300px;
        overflow-y: scroll;
        border: 1px solid #ccc;
        padding: 10px;
        background: #f9f9f9;
        font-family: monospace;
        white-space: pre-wrap;
    }
""") as ui:
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=300, type="messages")
        with gr.Column(scale=1):
            
            logs_box = gr.HTML(label="Logs", elem_id="log_box")
    with gr.Row():
        Model = gr.Dropdown(["Open AI","XX"],
                              # value=["Open AI","Claude"],
                              multiselect=False,
                              label="Model",
                              interactive=True)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")


    timer = gr.Timer(value=5, active=True)
    timer.tick(get_html_logs, inputs=None, outputs=[logs_box])

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        lf.logging.info(f"User message: {message}")
        yield "", history, get_html_logs()
        
    def set_initial_prompt():
        return [{"role": "assistant", "content": initial_prompt}]
    
    ui.load(set_initial_prompt, inputs=None, outputs=chatbot)
    
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot, logs_box]).then(
        chat, inputs=[chatbot, Model], outputs=[chatbot]
    )
    
    clear.click(clear_logs, inputs=None, outputs=[chatbot, logs_box], queue=False)