# imports
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import json
from classes import embeddings
from classes import log_formatter

 # Initialize ChromaDB client
DB_PATH = "agile_process"
client = chromadb.PersistentClient(path=DB_PATH)

# Initialize embeddings
embeddings_model = embeddings.SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
collection_name = "process_docs"
collection = client.get_collection(name=collection_name)
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
gpt_model = "gpt-4o-mini"

# Example: Query the collection
def query_documents(query_text, n_results=10):
    """Query the document collection."""
    query_embedding = embeddings.embeddings_model.embed_query(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

system_message = """You devise the strategy on how to set up and start a new data engineering project from scratch. You ask the user
    to share details about source systems, the number of final data marts, and a brief outline of the overall architecture. Do not proceed 
    unless the user shares the information related to overall architecture, details about source systems, and the number of target data
    marts. On getting the aforesaid information from the user, you will make the tool call using the return_context_function and strictly
    adhere to the agile process details that the tool call returns. You will eventually display the epics, features, and user stories 
    needed to achieve the end objective, which is to create the final data marts. Always provide epic, feature, and story details and 
    include points for each story along with the output."""



return_context_function = {
    "name": "return_context",
    "description": """Call this tool after the user confirms the details about source systems, the number of target data marts, and the 
    overall architecture. For any details related to agile processes, you will strictly adhere to the output that you gather from this 
    tool call.""",
    "parameters": {
        "type": "object",
        "properties": {
            "details_source_systems": {
                "type": "string",
                "description": "The details about source systems",
            },
            "overall_architecture": {
                "type": "string",
                "description": "Details about the architecture",
            },
            "target_data_marts": {
                "type": "string",
                "description": "Details about the target data marts",
            }
        }
    },
    "required": ["details_source_systems","overall_architecture","target_data_marts"],
    "additionalProperties": False
}

tools = [{"type": "function", "function": return_context_function}]

# We have to write that function handle_tool_call:

def handle_tool_call(name, args):
    source = args.get('details_source_systems')
    architecture = args.get('overall_architecture')
    marts = args.get('target_data_marts')
    if name.replace('"','') == "return_context":
        context=return_context(collection, f"Source details -\n{source}\n\nArchitecture Details -\n{architecture}\n\nMart Details -\n{marts}")
            
    return context

def return_context(collection, user_query):
    context = "\n\nProviding some context from relevant information -\n\n"
    retrieved = collection.query(
        query_embeddings=[embeddings_model.embed_query(user_query)],
        n_results=10,  # e.g., 5 or 10
        include=["documents", "metadatas"]
    )
    retrieved_chunks = retrieved["documents"][0]
    context+= "\n\n".join(retrieved_chunks)
    return context

def chat_open_ai(history):
    openai=OpenAI()
    messages = [{"role": "system", "content": system_message}] + history 
    response = openai.chat.completions.create(model=gpt_model, messages=messages, tools=tools)

    tool_responses = []

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls  # renamed to avoid UnboundLocalError

        print(f"tool calls \n\n {tool_calls}")

        for tool_call in tool_calls:
            tool_id = tool_call.id
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Call the tool handler
            result = ""
            if name == "return_context":
                result = handle_tool_call(name, args)

            tool_responses.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": json.dumps(result),
            })

        print(f"tool responses {tool_responses}")
        messages.append(message)
        messages.extend(tool_responses)  # important fix here

        response = openai.chat.completions.create(
            model=gpt_model,
            messages=messages,
            tools=tools
        )

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]

    return history