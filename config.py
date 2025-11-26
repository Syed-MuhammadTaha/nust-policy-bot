import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

class Config:
    """Centralized configuration settings for the RAG chatbot with All-Cloud Hybrid Retrieval."""

    # Secret API Keys
    JINA_API_KEY = os.getenv("JINA_API_KEY")

    # Embedding Models Configuration (All Jina Cloud)
    # Dense Embeddings - Jina Cloud API
    JINA_EMBEDDING_MODEL = "jina-embeddings-v2-base-en"
    DENSE_VECTOR_NAME = "jina-dense"
    DENSE_DIMENSION = 768  # Jina v2 base model dimension
    
    # Sparse Embeddings - Jina Cloud API (same model, different output)
    SPARSE_VECTOR_NAME = "jina-sparse"
    
    # Reranking - Jina Reranker Cloud API
    JINA_RERANKER_MODEL = "jina-reranker-v2-base-multilingual"

    # Qdrant Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hybrid-search")
    QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    # Paths
    UPLOAD_FOLDER = "data"
    PROCESSED_FILES_PATH = "qdrant_metadata/processed_files.json"
    
    # Chunking Configuration
    CHUNKING_STRATEGY = "semantic"  # Options: "semantic", "fixed"
    CHUNK_SIZE = 800  # Larger chunks for better context
    CHUNK_OVERLAP = 200  # Substantial overlap for continuity
    
    # Retrieval Configuration
    PREFETCH_LIMIT = 20  # Number of results from each sub-query (dense + sparse)
    RERANK_LIMIT = 10    # Number of candidates to send to reranker
    FINAL_LIMIT = 5      # Final number of results after reranking

    @staticmethod
    def display():
        """Display current configuration (excluding secrets)."""
        print(f"🔹 Embedding Model: {Config.JINA_EMBEDDING_MODEL} (Jina Cloud - Dense & Sparse)")
        print(f"🔹 Reranker Model: {Config.JINA_RERANKER_MODEL} (Jina Cloud)")
        print(f"🔹 Qdrant URL: {Config.QDRANT_URL}")
        print(f"🔹 Collection Name: {Config.QDRANT_COLLECTION_NAME}")
        print(f"🔹 Upload Folder: {Config.UPLOAD_FOLDER}")
        print(f"🔹 Chunking Strategy: {Config.CHUNKING_STRATEGY}")
        print(f"🔹 Chunk Size: {Config.CHUNK_SIZE}")
        print(f"🔹 Chunk Overlap: {Config.CHUNK_OVERLAP}")
        print(f"🔹 Prefetch Limit: {Config.PREFETCH_LIMIT}")
        print(f"🔹 Rerank Candidates: {Config.RERANK_LIMIT}")
        print(f"🔹 Final Results: {Config.FINAL_LIMIT}")
