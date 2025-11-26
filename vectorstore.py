import os
import json
import uuid
from typing import List, Tuple, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, models
from langchain_community.embeddings import JinaEmbeddings
from config import Config
from preprocessing_simple import preprocess_pdf_simple

# Initialize Jina Embeddings (Cloud API)
print("🔄 Loading Jina Cloud API...")
print(f"   ☁️  Model: {Config.JINA_EMBEDDING_MODEL}")

jina_embeddings = JinaEmbeddings(
    jina_api_key=Config.JINA_API_KEY,
    model_name=Config.JINA_EMBEDDING_MODEL
)

print("✅ Jina Cloud API ready!")
print(f"   ☁️  Dense embeddings via Jina Cloud")
print(f"   ☁️  Sparse/BM25 via Qdrant built-in text indexing")
print(f"   ☁️  Reranking via Jina Reranker API")

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    host=Config.QDRANT_HOST,
    port=Config.QDRANT_PORT
)

PROCESSED_FILES_PATH = Config.PROCESSED_FILES_PATH


def ensure_collection_exists():
    """
    Ensure the Qdrant collection exists with hybrid search configuration.
    Uses Jina Cloud for dense embeddings + Qdrant's built-in BM25 for sparse matching.
    """
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if Config.QDRANT_COLLECTION_NAME not in collection_names:
            # Get sample embedding to determine dimensions
            sample_text = ["sample"]
            sample_dense = jina_embeddings.embed_documents(sample_text)[0]
            
            qdrant_client.create_collection(
                collection_name=Config.QDRANT_COLLECTION_NAME,
                vectors_config={
                    # Dense embeddings from Jina Cloud for semantic search
                    Config.DENSE_VECTOR_NAME: models.VectorParams(
                        size=len(sample_dense),
                        distance=models.Distance.COSINE,
                    ),
                },
                # Enable full-text search for BM25-style keyword matching
                # Qdrant will index the text payload automatically
            )
            print(f"✅ Created hybrid search collection: {Config.QDRANT_COLLECTION_NAME}")
            print(f"   ☁️  Dense vectors (Jina Cloud): {Config.DENSE_VECTOR_NAME} ({len(sample_dense)} dims)")
            print(f"   ☁️  Text indexing (Qdrant BM25): Enabled on 'document' field")
        return True
    except Exception as e:
        print(f"❌ Error ensuring collection exists: {e}")
        return False


def get_processed_files() -> Dict:
    """Returns a dict of processed files with their metadata."""
    os.makedirs(os.path.dirname(PROCESSED_FILES_PATH) or '.', exist_ok=True)
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_processed_files(processed_files: Dict):
    """Saves the processed files metadata to disk."""
    os.makedirs(os.path.dirname(PROCESSED_FILES_PATH) or '.', exist_ok=True)
    with open(PROCESSED_FILES_PATH, 'w') as f:
        json.dump(processed_files, f, indent=2)


def is_file_processed(file_path: str) -> bool:
    """Check if a file has already been processed."""
    processed_files = get_processed_files()
    file_name = os.path.basename(file_path)
    
    if file_name not in processed_files:
        return False
    
    # Check if file has been modified since last processing
    current_mtime = os.path.getmtime(file_path)
    stored_mtime = processed_files[file_name].get('mtime', 0)
    
    return current_mtime == stored_mtime


def process_and_store(file_path: str, force_reprocess: bool = False, chunking_strategy: str = None) -> Tuple[bool, str, int]:
    """
    Loads document and adds to Qdrant with Jina Cloud embeddings.
    Uses:
    - Jina Cloud API for dense embeddings
    - Qdrant's built-in text indexing for BM25/sparse matching
    - Jina Reranker API for reranking (applied at query time)
    
    Args:
        file_path: Path to the PDF file
        force_reprocess: If True, reprocess even if file was already processed
        chunking_strategy: Strategy to use - "semantic" or "fixed" (defaults to Config value)
    
    Returns:
        tuple: (success: bool, message: str, chunks_added: int)
    """
    file_name = os.path.basename(file_path)
    
    # Ensure collection exists
    if not ensure_collection_exists():
        return False, "Failed to connect to Qdrant or create collection", 0
    
    # Check if already processed
    if not force_reprocess and is_file_processed(file_path):
        return True, f"File '{file_name}' already processed. Skipping.", 0
    
    # Configure chunking from Config or override
    strategy = chunking_strategy or Config.CHUNKING_STRATEGY
    
    try:
        # Use simplified but effective preprocessing
        docs = preprocess_pdf_simple(
            file_path,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            strategy=strategy
        )
        
        print(f"📄 Processing {len(docs)} chunks from '{file_name}'...")
        
        # Extract text content for embedding
        documents = [doc.page_content for doc in docs]
        
        # Generate dense embeddings via Jina Cloud
        print("☁️  Generating dense embeddings via Jina Cloud API...")
        dense_embeddings = jina_embeddings.embed_documents(documents)
        
        # Create points with embeddings and full text for BM25
        points = []
        for idx, (doc, dense_emb) in enumerate(zip(docs, dense_embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generate unique UUID for each point
                vector={
                    Config.DENSE_VECTOR_NAME: dense_emb,
                },
                payload={
                    "document": doc.page_content,  # Qdrant will index this for BM25
                    "source": doc.metadata.get('source', file_name),
                    "page": doc.metadata.get('page', 'N/A'),
                    "chunk_title": doc.metadata.get('chunk_title', ''),
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        print(f"⬆️  Uploading {len(points)} points to Qdrant...")
        operation_info = qdrant_client.upsert(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            points=points
        )
        
        # Update processed files metadata
        processed_files = get_processed_files()
        processed_files[file_name] = {
            'mtime': os.path.getmtime(file_path),
            'chunks': len(docs),
            'chunking_strategy': strategy,
            'chunk_size': Config.CHUNK_SIZE,
            'chunk_overlap': Config.CHUNK_OVERLAP,
            'processed_at': str(os.path.getmtime(file_path)),
            'embedding_source': 'Jina Cloud API',
            'search_methods': ['Dense (Jina)', 'BM25 (Qdrant)', 'Rerank (Jina)']
        }
        save_processed_files(processed_files)
        
        print(f"✅ Successfully processed '{file_name}'")
        return True, f"Successfully processed '{file_name}' using {strategy} strategy with Jina Cloud embeddings", len(docs)
        
    except Exception as e:
        print(f"❌ Error processing '{file_name}': {str(e)}")
        return False, f"Error processing '{file_name}': {str(e)}", 0


def process_all_pdfs_in_folder(folder_path: str) -> Dict:
    """
    Process all PDF files in the specified folder with Jina Cloud embeddings.
    
    Args:
        folder_path: Path to folder containing PDFs
    
    Returns:
        dict: Summary of processing results
    """
    if not os.path.exists(folder_path):
        return {"error": f"Folder '{folder_path}' does not exist"}
    
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return {"message": "No PDF files found in folder", "processed": 0}
    
    results = {
        "total_files": len(pdf_files),
        "newly_processed": 0,
        "already_processed": 0,
        "failed": 0,
        "total_chunks": 0,
        "files": []
    }
    
    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        try:
            success, message, chunks = process_and_store(file_path)
            
            if "already processed" in message:
                results["already_processed"] += 1
            elif success:
                results["newly_processed"] += 1
                results["total_chunks"] += chunks
            else:
                results["failed"] += 1
            
            results["files"].append({
                "name": pdf_file,
                "status": "success" if success else "failed",
                "message": message,
                "chunks": chunks
            })
        except Exception as e:
            results["failed"] += 1
            results["files"].append({
                "name": pdf_file,
                "status": "failed",
                "message": str(e)
            })
    
    return results


def get_vectorstore_stats() -> Dict:
    """Get statistics about the hybrid search vector store."""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if Config.QDRANT_COLLECTION_NAME not in collection_names:
            return {"exists": False}
        
        # Get collection info
        collection_info = qdrant_client.get_collection(Config.QDRANT_COLLECTION_NAME)
        total_points = collection_info.points_count
        
        # Get processed files metadata
        processed_files = get_processed_files()
        
        return {
            "exists": True,
            "total_files": len(processed_files),
            "total_chunks": total_points,
            "files": list(processed_files.keys()),
            "search_methods": [
                "Dense Semantic (Jina Cloud)",
                "BM25 Keyword (Qdrant)",
                "Reranking (Jina Cloud)"
            ]
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {"exists": False, "error": str(e)}


# Export the client and embeddings for use in rag_chain.py
__all__ = [
    'qdrant_client',
    'jina_embeddings',
    'process_and_store',
    'process_all_pdfs_in_folder',
    'get_vectorstore_stats',
    'is_file_processed'
]
