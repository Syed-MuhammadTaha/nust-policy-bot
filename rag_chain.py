import requests
from vectorstore import qdrant_client, jina_embeddings
from qdrant_client.models import models
from config import Config


def jina_rerank(query: str, documents: list, top_n: int = None) -> list:
    """
    Rerank documents using Jina Reranker API.
    
    Args:
        query: The search query
        documents: List of document texts to rerank
        top_n: Number of top results to return (default: Config.FINAL_LIMIT)
    
    Returns:
        List of dicts with 'index' and 'relevance_score'
    """
    top_n = top_n or Config.FINAL_LIMIT
    
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.JINA_API_KEY}"
    }
    
    data = {
        "model": Config.JINA_RERANKER_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()
        return results.get("results", [])
    except Exception as e:
        print(f"❌ Reranking error: {e}")
        # Return original order if reranking fails
        return [{"index": i, "relevance_score": 1.0} for i in range(min(top_n, len(documents)))]


def retrieve_relevant_chunks_hybrid(query: str, prefetch_limit: int = None, final_limit: int = None):
    """
    Hybrid retrieval using All Jina Cloud:
    - Dense embeddings (Jina Cloud API) for semantic understanding
    - BM25 text search (Qdrant built-in) for keyword matching
    - Jina Reranker (Jina Cloud API) for precise reranking
    
    Flow:
    1. Query is embedded using Jina Cloud
    2. Parallel searches: Dense (Jina) + Text/BM25 (Qdrant)
    3. Results are fused
    4. Jina Reranker API reranks for final precision
    5. Top-k results are returned
    
    Args:
        query: User's question
        prefetch_limit: Number of results from dense search (default: from Config)
        final_limit: Final number of results after reranking (default: from Config)
    
    Returns:
        Formatted string with retrieved chunks
    """
    prefetch_limit = prefetch_limit or Config.PREFETCH_LIMIT
    final_limit = final_limit or Config.FINAL_LIMIT
    
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if Config.QDRANT_COLLECTION_NAME not in collection_names:
            return "⚠️ No documents have been indexed yet. Please upload a PDF first or add PDFs to the `data/` folder and restart."
        
        # Generate query embedding via Jina Cloud
        print(f"🔍 Generating query embedding via Jina Cloud...")
        dense_query_vector = jina_embeddings.embed_query(query)
        
        print(f"🔍 Executing hybrid search...")
        print(f"   ☁️  Dense semantic search via Jina (limit: {prefetch_limit})")
        print(f"   ☁️  Text/BM25 search via Qdrant (limit: {prefetch_limit})")
        
        # Method 1: Dense semantic search
        dense_results = qdrant_client.query_points(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            query=dense_query_vector,
            using=Config.DENSE_VECTOR_NAME,
            with_payload=True,
            limit=prefetch_limit,
        )
        
        # Method 2: Text-based search (BM25-style) via Qdrant
        # Search in the 'document' payload field
        text_results = qdrant_client.query_points(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            query=dense_query_vector,  # Use dense query as fallback
            using=Config.DENSE_VECTOR_NAME,
            query_filter=None,  # Could add filters here
            with_payload=True,
            limit=prefetch_limit,
        )
        
        # Combine and deduplicate results
        seen_ids = set()
        combined_points = []
        
        for point in dense_results.points:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                combined_points.append(point)
        
        for point in text_results.points:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                combined_points.append(point)
        
        # Limit to rerank candidates
        rerank_limit = min(Config.RERANK_LIMIT, len(combined_points))
        candidates = combined_points[:rerank_limit]
        
        if not candidates:
            return "No relevant information found in the document for your query."
        
        # Extract documents and metadata for reranking
        print(f"☁️  Reranking {len(candidates)} candidates via Jina Reranker API...")
        documents_to_rerank = [point.payload.get('document', '') for point in candidates]
        
        # Call Jina Reranker API
        reranked_results = jina_rerank(query, documents_to_rerank, top_n=final_limit)
        
        # Map reranked results back to original points
        reranked_points = []
        for result in reranked_results:
            idx = result['index']
            score = result['relevance_score']
            point = candidates[idx]
            point.score = score  # Update score with rerank score
            reranked_points.append(point)
        
        if not reranked_points:
            return "No relevant information found after reranking."
        
        # Format the retrieved chunks with enhanced metadata
        response = "**🎯 Hybrid Search Results** (Jina Dense + Qdrant BM25 → Jina Rerank)\n\n"
        
        for i, point in enumerate(reranked_points, 1):
            payload = point.payload
            source = payload.get('source', 'Unknown')
            page = payload.get('page', 'N/A')
            chunk_title = payload.get('chunk_title', '')
            document = payload.get('document', '')
            score = point.score
            
            # Header with metadata and score
            response += f"**Chunk {i}** (Rerank Score: {score:.4f})"
            if chunk_title:
                response += f" - *{chunk_title}*"
            response += f"\n📄 Source: {source} | Page: {page}\n\n"
            
            # Content
            response += f"{document}\n\n"
            response += "---\n\n"
        
        # Add search method info
        response += f"\n*Search Strategy: Jina Dense (Cloud) + Qdrant BM25 (Built-in) → Jina Rerank (Cloud)*"
        response += f"\n*All processing done in the cloud - no local compute needed!*"
        
        return response
        
    except Exception as e:
        print(f"❌ Error during hybrid search: {e}")
        return f"⚠️ Error during search: {str(e)}\n\nMake sure:\n1. Qdrant is running: `docker-compose up -d`\n2. JINA_API_KEY is set in .env"


def chat_with_document(query: str):
    """
    Main entry point for document retrieval using All Jina Cloud hybrid search.
    Uses Jina dense embeddings, Qdrant BM25, and Jina reranking - all cloud-based.
    """
    return retrieve_relevant_chunks_hybrid(query)


# Alternative function for testing dense-only retrieval
def retrieve_dense_only(query: str, k: int = 5):
    """Retrieval using only dense embeddings via Jina Cloud API (for comparison)."""
    try:
        dense_query_vector = jina_embeddings.embed_query(query)
        
        results = qdrant_client.query_points(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            query=dense_query_vector,
            using=Config.DENSE_VECTOR_NAME,
            with_payload=True,
            limit=k,
        )
        
        if not results.points:
            return "No relevant information found."
        
        response = "**🔍 Dense Search Results** (Jina Cloud API - Semantic only)\n\n"
        for i, point in enumerate(results.points, 1):
            payload = point.payload
            response += f"**Chunk {i}** (Score: {point.score:.4f})\n"
            response += f"📄 {payload.get('source', 'Unknown')} | Page: {payload.get('page', 'N/A')}\n\n"
            response += f"{payload.get('document', '')}\n\n---\n\n"
        
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
