# imports
import gradio as gr
from agents.log_formatter_agent import LogFormatterAgent
from agents.frontier_agent import FrontierAgent
from agents.open_source_agent import OpenSourceAgent
from agents.RAG_agent import RAGAgent

lf=LogFormatterAgent()
lf.init_logging()

ra=RAGAgent()
fa=FrontierAgent(ra)
oa=OpenSourceAgent(ra)

def get_html_logs():
    lf.html_log_stream.seek(0)
    return lf.html_log_stream.read()

def clear_logs():
    lf.log_stream.truncate(0)
    lf.log_stream.seek(0)
    lf.html_log_stream.truncate(0)
    lf.html_log_stream.seek(0)
    return None, ""

def chat(history,Model):
    if Model=="Open AI (gpt-4o-mini)":
        history = fa.chat_open_ai(history)
    elif Model=="Open Source (HuggingFace Llama-3.1)":
        history = oa.chat_llama(history)
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
        Model = gr.Dropdown(["Open AI (gpt-4o-mini)","Open Source (HuggingFace Llama-3.1)"],
                              multiselect=False,
                              label="Model",
                              interactive=True)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")


    timer = gr.Timer(value=2, active=True)
    timer.tick(get_html_logs, inputs=None, outputs=[logs_box])

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        lf.logger.info(f"User message: {message}")
        yield "", history, get_html_logs()
        
    def set_initial_prompt():
        return [{"role": "assistant", "content": initial_prompt}]
    
    ui.load(set_initial_prompt, inputs=None, outputs=chatbot)
    
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot, logs_box]).then(
        chat, inputs=[chatbot, Model], outputs=[chatbot]
    )
    
    clear.click(clear_logs, inputs=None, outputs=[chatbot, logs_box], queue=False)

    ui.launch(inbrowser=True)