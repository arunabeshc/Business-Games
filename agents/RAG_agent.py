import chromadb
from dotenv import load_dotenv
import chromadb
import logging
from classes import embeddings

class RAGAgent:

    def __init__(self):
            
        self.logger = logging.getLogger("RAGAgent")
        # Initialize ChromaDB client
        DB_PATH = "agile_process"
        client = chromadb.PersistentClient(path=DB_PATH)

        # Initialize embeddings
        self.embeddings_model = embeddings.SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
        collection_name = "process_docs"
        self.collection = client.get_collection(name=collection_name)
        load_dotenv(override=True)
        self.logger.info("RAG Agent is ready")


    def return_context(self, collection, user_query, n_results):
        context = "\n\nProviding some context from relevant information -\n\n"
        retrieved = collection.query(
            query_embeddings=[self.embeddings_model.embed_query(user_query)],
            n_results=n_results,  # e.g., 5 or 10
            include=["documents", "metadatas"]
        )
        retrieved_chunks = retrieved["documents"][0]
        context+= "\n\n".join(retrieved_chunks)
        self.logger.info(f"RAG Agent is providing context from the Vector Database")
        return context