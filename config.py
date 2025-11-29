import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

class Config:
    """Centralized configuration settings for the RAG chatbot with All-Cloud Hybrid Retrieval."""

    # Secret API Keys
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Embedding Models Configuration (All Jina Cloud)
    # Dense Embeddings - Jina Cloud API
    JINA_EMBEDDING_MODEL = "jina-embeddings-v2-base-en"
    DENSE_VECTOR_NAME = "jina-dense"
    DENSE_DIMENSION = 768  # Jina v2 base model dimension
    
    # Sparse Embeddings - Jina Cloud API (same model, different output)
    SPARSE_VECTOR_NAME = "jina-sparse"
    
    # Reranking - Jina Reranker Cloud API
    JINA_RERANKER_MODEL = "jina-reranker-v2-base-multilingual"
    
    # Generation Model - Google Gemini
    GOOGLE_MODEL = "gemini-2.5-flash"  # Fast and efficient
    GENERATION_TEMPERATURE = 0  # Very low temperature for controlled, factual responses
    GENERATION_MAX_TOKENS = 1024  # Response length limit

    # Qdrant Configuration
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hybrid-search")
    QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    # Paths
    UPLOAD_FOLDER = "data"
    PROCESSED_FILES_PATH = "qdrant_metadata/processed_files.json"
    
    # PDF Source URLs for Grounded Citations
    # Map PDF filenames to their online URLs
    PDF_SOURCE_URLS = {
        "Code_of_Ethics.pdf":"https://nust.edu.pk/wp-content/uploads/2020/03/Code_of_Ethics.pdf",
        "Protection-against-Harassment-of-Women.pdf":"https://nust.edu.pk/wp-content/uploads/2020/03/Protection-against-Harassment-of-Women.pdf",
        "NUST-Fee-Policy-Salient-Features.pdf":"https://nust.edu.pk/wp-content/uploads/2021/09/NUST-Fee-Policy-Salient-Features.pdf",
        "POLICY-ON-NUST-STUDENT-FORUM-NSF-Student-Copy.pdf":"https://sa.nust.edu.pk/wp-content/uploads/2024/05/POLICY-ON-NUST-STUDENT-FORUM-NSF-Student-Copy.pdf",
        "Inbound_policy.pdf":"https://nust.edu.pk/wp-content/uploads/2025/10/552978584685WP_-_Inbound_-_Web.pdf",
        "DinningPolicies.pdf":"https://nust.edu.pk/wp-content/uploads/2024/06/MARCOMS-217-Dress-Norms-Dinning-Etiquette-V.7.0-24062024.pdf",
    }
    
    # Citation format preferences
    ENABLE_CITATIONS = True
    ENABLE_PAGE_LINKS = True  # Include page numbers in citations
    ENABLE_SMART_HIGHLIGHTING = True  # Use semantic matching to find relevant sentences for PDF highlighting
    CITATION_STYLE = "inline"  # Options: "inline", "footnote"
    
    # Chunking Configuration
    CHUNKING_STRATEGY = "fixed"  # Fixed chunking works consistently across all document types
    CHUNK_SIZE = 1200  # Balanced size for complete context without being too large
    CHUNK_OVERLAP = 250  # Good overlap to maintain context between chunks
    
    # Retrieval Configuration
    PREFETCH_LIMIT = 10  # Number of results from each sub-query (dense + sparse)
    RERANK_LIMIT = 6     # Number of candidates to send to reranker
    FINAL_LIMIT = 3      # Final number of results after reranking
    
    # Abstention Parameters (prevents answering off-topic questions)
    MIN_RELEVANCE_SCORE = 0.30  # Minimum rerank score to consider relevant (0-1 scale)
    # If best chunk score < this threshold, abstain from answering

    @staticmethod
    def display():
        """Display current configuration (excluding secrets)."""
        print(f"🔹 Embedding Model: {Config.JINA_EMBEDDING_MODEL} (Jina Cloud)")
        print(f"🔹 Reranker Model: {Config.JINA_RERANKER_MODEL} (Jina Cloud)")
        print(f"🔹 Generation Model: {Config.GOOGLE_MODEL} (Google Gemini)")
        print(f"🔹 Temperature: {Config.GENERATION_TEMPERATURE}")
        print(f"🔹 Qdrant URL: {Config.QDRANT_URL}")
        print(f"🔹 Collection Name: {Config.QDRANT_COLLECTION_NAME}")
        print(f"🔹 Upload Folder: {Config.UPLOAD_FOLDER}")
        print(f"🔹 Chunking Strategy: {Config.CHUNKING_STRATEGY}")
        print(f"🔹 Chunk Size: {Config.CHUNK_SIZE}")
        print(f"🔹 Chunk Overlap: {Config.CHUNK_OVERLAP}")
        print(f"🔹 Prefetch Limit: {Config.PREFETCH_LIMIT}")
        print(f"🔹 Rerank Candidates: {Config.RERANK_LIMIT}")
        print(f"🔹 Final Results: {Config.FINAL_LIMIT}")
