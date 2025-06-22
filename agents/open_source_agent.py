# imports
from dotenv import load_dotenv
import json
import logging
import modal

class OpenSourceAgent:

    def __init__(self,ra):
        
        self.logger = logging.getLogger("OpenSourceAgent")
        self.ra=ra
        self.n_results=1
        self.logger.info("Open Source Agent is initializing - connecting to modal")
        self.llama = modal.Function.from_name("llama_3.1_assistant", "generate")
        self.system_message = """You devise the strategy on how to set up and start a new data engineering project 
        from scratch. You ask the user to share details about source systems, the number of final data marts, and 
        a brief outline of the overall architecture. Do not proceed unless the user shares the information related 
        to overall architecture, details about source systems, and the number of target data marts. On getting the 
        aforesaid information from the user, based on the input provided, you will eventually display the epics, 
        features, and user stories needed to achieve the end objective, which is to create the final data marts. 
        Always provide epic, feature, and story details and include points for each story along with the output. 
        Give as detailed output as possible."""
        self.logger.info("Open Source Agent is ready")

    def messages_to_text(self, messages):
        role_map = {
            "system": "[System]",
            "user": "[User]",
            "assistant": "[Assistant]"
        }
        return "\n".join(f"{role_map.get(msg['role'], '[Unknown]')}: {msg['content']}" for msg in messages)

    def update_last_user_message(self, history, new_content):
        for i in reversed(range(len(history))):
            if history[i]["role"] == "user":
                history[i]["content"] += "\n" + new_content
                break  # Only update the last one
        return history

    def chat_llama(self, history):
        messages = [{"role": "system", "content": self.system_message}] + history 
        self.logger.info("Open Source Agent is getting context from RAG agent")
        context=self.ra.return_context(self.ra.collection,self.messages_to_text(messages),self.n_results)
        messages=self.update_last_user_message(messages,context)
        self.logger.info("Open Source Agent is calling Llama 3.1 Model")
        reply = self.llama.remote(messages)
        self.logger.info("Open Source Agent is returning the response")
        history += [{"role": "assistant", "content": reply}]
        return history