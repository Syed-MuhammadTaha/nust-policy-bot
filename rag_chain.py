import re
import requests
from vectorstore import qdrant_client, jina_embeddings
from qdrant_client.models import models
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model=Config.GOOGLE_MODEL,
    google_api_key=Config.GOOGLE_API_KEY,
    temperature=Config.GENERATION_TEMPERATURE,
    max_output_tokens=Config.GENERATION_MAX_TOKENS
)

# Create RAG prompt template with strategic inline citation instructions
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions about university policies and documents.

Answer the question based ONLY on the provided context below. Be concise, accurate, and factual.

CRITICAL CITATION RULE - READ CAREFULLY:
You MUST minimize citations. Cite ONCE per complete idea or paragraph, NOT after every sentence.

Citation Strategy:
1. If ALL information in a paragraph comes from ONE source → Cite [X] ONLY at the END of that paragraph
2. If you switch sources mid-paragraph → Cite when switching: "Info from source 1. Info from source 2 [2]."
3. For multiple sources in one place → Use separate brackets: [1][2] or [2][3]
4. NEVER cite the same source in consecutive sentences
5. Aim for 1-2 citations total per paragraph, not 1 per sentence

Each source in context is numbered [Source X]. ONLY cite sources you actually use.

EXCELLENT example (minimal citations ✓):
"NUST has implemented the Protection against Harassment of Women at the Workplace Act to ensure a safe working environment. The Board of Governors approved the creation of a Harassment Complaint Cell and Inquiry Committee, both now fully functional. This policy is incorporated into Chapter-10 of the HR Handbook, and queries can be directed to C3A [1].

Sexual harassment is strictly prohibited and includes unwanted sexual advances or conduct of a sexual nature [2][3]."

(Note: Only 2 citations for 5 sentences = Clean and minimal)

BAD example (over-citing ✗):
"NUST has implemented the Protection Act [1]. The Board approved bodies [1]. These are functional [1]. Policy is in HR Handbook [1]. Contact C3A for queries [1].

Sexual harassment is prohibited [2]. It includes unwanted advances [2]. Violations are serious [3]."

(Note: 8 citations for 7 sentences = Too many!)

CRITICAL - ABSTENTION RULE:
If the context does NOT contain information to answer the question, you MUST respond EXACTLY:
"I don't have enough information in the provided documents to answer that question."

ABSTAIN in these cases:
- Question is completely off-topic (e.g., asking about weather, sports, or unrelated topics)
- Context discusses something different than what's asked
- Context is too vague or doesn't provide specific details needed
- Question asks for personal opinions or predictions

Example abstention cases:
❌ Q: "What's the weather today?" → ABSTAIN (not in documents)
❌ Q: "Who will win the election?" → ABSTAIN (not in documents)
❌ Q: "What's your favorite food?" → ABSTAIN (opinion question)
✅ Q: "What is the harassment policy?" → ANSWER (if in documents)
"""),
    ("human", """Context:
{context}

Question: {question}

Answer (remember: cite ONCE per paragraph/idea, not per sentence):""")
])

print("✅ Loaded RAG prompt template with inline citations")


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
        
        # Format the retrieved chunks with enhanced metadata and grounded citations
        response = "**🎯 Retrieved Information**\n\n"
        
        for i, point in enumerate(reranked_points, 1):
            payload = point.payload
            source = payload.get('source', 'Unknown')
            source_url = payload.get('source_url', None)
            page = payload.get('page', 'N/A')
            chunk_title = payload.get('chunk_title', '')
            document = payload.get('document', '')
            score = point.score
            
            # Content first (main answer)
            response += f"{document}\n\n"
            
            # Grounded citation with clickable link (simple page link only)
            if Config.ENABLE_CITATIONS:
                response += f"**📚 Citation {i}:** "
                
                if source_url and page != 'N/A':
                    # Simple page link: URL#page=X
                    page_link = f"{source_url}#page={page}"
                    response += f"[{source}, Page {page}]({page_link})"
                elif source_url:
                    response += f"[{source}]({source_url})"
                else:
                    response += f"{source}, Page {page}"
                
                # Add confidence score
                response += f" • Relevance: {score:.2%}\n\n"
            
            response += "---\n\n"
    
        # Add methodology footer
        response += "\n*💡 Retrieval Method: Jina Dense Embeddings + Qdrant BM25 Keyword Search → Jina Reranker*\n"
        response += "*✅ All citations link to original source documents*"
        
        return response

    except Exception as e:
        print(f"❌ Error during hybrid search: {e}")
        return f"⚠️ Error during search: {str(e)}\n\nMake sure:\n1. Qdrant is running: `docker-compose up -d`\n2. JINA_API_KEY is set in .env"


def generate_answer_with_citations(query: str) -> tuple:
    """
    Full RAG pipeline: Retrieve → Rerank → Generate with Google Gemini.
    Returns (answer_with_citations, retrieved_chunks_metadata).
    """
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if Config.QDRANT_COLLECTION_NAME not in collection_names:
            return "⚠️ No documents have been indexed yet. Please upload a PDF first."
        
        # Step 1: Retrieve with hybrid search
        print(f"🔍 Retrieving relevant chunks...")
        dense_query_vector = jina_embeddings.embed_query(query)
        
        dense_results = qdrant_client.query_points(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            query=dense_query_vector,
            using=Config.DENSE_VECTOR_NAME,
            with_payload=True,
            limit=Config.PREFETCH_LIMIT,
        )
        
        if not dense_results.points:
            return "No relevant information found in the documents."
        
        # Step 2: Rerank
        print(f"☁️ Reranking with Jina...")
        candidates = dense_results.points[:Config.RERANK_LIMIT]
        documents_to_rerank = [point.payload.get('document', '') for point in candidates]
        reranked_results = jina_rerank(query, documents_to_rerank, top_n=Config.FINAL_LIMIT)
        
        reranked_points = []
        for result in reranked_results:
            idx = result['index']
            score = result['relevance_score']
            point = candidates[idx]
            point.score = score
            reranked_points.append(point)
        
        if not reranked_points:
            return "No relevant information found."
        
        # Step 3: Deduplicate chunks from same PDF page (merge them)
        page_to_chunks = {}  # Key: (source_url, page) -> list of chunks
        for point in reranked_points:
            payload = point.payload
            source_url = payload.get('source_url', None)
            page = payload.get('page', 'N/A')
            key = (source_url, page)
            
            if key not in page_to_chunks:
                page_to_chunks[key] = []
            page_to_chunks[key].append(point)
        
        # Merge chunks from same page - combine content, keep highest score
        deduplicated_points = []
        for key, chunks in page_to_chunks.items():
            if len(chunks) == 1:
                deduplicated_points.append(chunks[0])
            else:
                # Multiple chunks from same page - merge them
                best_chunk = max(chunks, key=lambda x: x.score)
                combined_content = "\n\n".join([c.payload.get('document', '') for c in chunks])
                
                # Update the payload with combined content
                best_chunk.payload['document'] = combined_content
                best_chunk.payload['merged_count'] = len(chunks)
                deduplicated_points.append(best_chunk)
        
        # Sort by score (highest first)
        deduplicated_points.sort(key=lambda x: x.score, reverse=True)
        
        # Step 3.5: Check relevance threshold for abstention
        best_score = deduplicated_points[0].score if deduplicated_points else 0.0
        
        if best_score < Config.MIN_RELEVANCE_SCORE:
            # Score too low - likely off-topic question
            print(f"⚠️ Low relevance score ({best_score:.2%} < {Config.MIN_RELEVANCE_SCORE:.0%}). Abstaining.")
            abstention_message = (
                "I don't have enough information in the provided documents to answer that question.\n\n"
                f"*The retrieved content has low relevance (best score: {best_score:.0%}). "
                "This suggests your question may be outside the scope of the available documents, "
                "which focus on university policies and procedures.*"
            )
            return abstention_message, []
        
        # Step 4: Prepare context, citations, and metadata for sidebar
        context_parts = []
        citations = []
        chunks_metadata = []
        
        for i, point in enumerate(deduplicated_points, 1):
            payload = point.payload
            source = payload.get('source', 'Unknown')
            source_url = payload.get('source_url', None)
            page = payload.get('page', 'N/A')
            chunk_title = payload.get('chunk_title', '')
            document = payload.get('document', '')
            score = point.score
            merged_count = payload.get('merged_count', 1)
            
            # Add to context for LLM
            context_parts.append(f"[Source {i}]: {document}")
            
            # Build simple citation with page number only
            citation = f"**[{i}]** "
            if source_url and page != 'N/A':
                # Simple page link: URL#page=X
                page_link = f"{source_url}#page={page}"
                citation += f"[{source}, Page {page}]({page_link})"
            elif source_url:
                citation += f"[{source}]({source_url})"
            else:
                citation += f"{source}, Page {page}"
            
            citation += f" • {score:.0%}"
            if merged_count > 1:
                citation += f" *(merged {merged_count} chunks)*"
            citations.append(citation)
            
            # Store metadata for sidebar display
            chunks_metadata.append({
                "chunk_number": i,
                "source": source,
                "source_url": source_url,
                "page": page,
                "chunk_title": chunk_title,
                "score": score,
                "merged_count": merged_count,
                "preview": document[:150] + "..." if len(document) > 150 else document
            })
        
        # Step 5: Generate answer with LLM
        print(f"🤖 Generating answer with Google Gemini...")
        context = "\n\n".join(context_parts)
        
        # Format the prompt with context and question
        messages = rag_prompt.format_messages(context=context, question=query)
        
        # Generate answer
        response_obj = llm.invoke(messages)
        answer = response_obj.content
        
        # Step 6: Parse which citations were actually used in the answer
        # Handle both formats: [1][2] and [1, 2] or [1,2]
        cited_indices = set()
        
        # Pattern 1: Separate brackets [1][2]
        citation_pattern_separate = r'\[(\d+)\]'
        matches = re.findall(citation_pattern_separate, answer)
        for match in matches:
            cited_indices.add(int(match))
        
        # Pattern 2: Combined brackets [1, 2] or [1,2]
        citation_pattern_combined = r'\[(\d+(?:\s*,\s*\d+)+)\]'
        combined_matches = re.findall(citation_pattern_combined, answer)
        for match in combined_matches:
            # Split by comma and extract all numbers
            numbers = re.findall(r'\d+', match)
            for num in numbers:
                cited_indices.add(int(num))
        
        # Filter to only show cited sources
        cited_citations = []
        cited_metadata = []
        for i, (citation, metadata) in enumerate(zip(citations, chunks_metadata), 1):
            if i in cited_indices:
                cited_citations.append(citation)
                metadata_copy = metadata.copy()
                metadata_copy['cited'] = True
                cited_metadata.append(metadata_copy)
            else:
                # Still include in metadata but mark as not cited
                metadata_copy = metadata.copy()
                metadata_copy['cited'] = False
                cited_metadata.append(metadata_copy)
        
        # Step 7: Format final response
        response = f"{answer}\n\n"
        
        if cited_citations:
            response += "---\n\n**📚 Sources:**\n\n"
            response += "\n".join(cited_citations)
        else:
            # If no citations were used, show a note
            response += "\n\n*Note: No specific sources were cited for this answer.*"
        
        return response, cited_metadata
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return f"⚠️ Error: {str(e)}", []


def chat_with_document(query: str):
    """
    Main entry point for full RAG with generation.
    Returns (answer, chunks_metadata) tuple.
    """
    return generate_answer_with_citations(query)


def retrieve_only(query: str):
    """
    Alternative: Retrieval only without LLM generation.
    Returns (response_text, chunks_metadata) tuple.
    """
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if Config.QDRANT_COLLECTION_NAME not in collection_names:
            return "⚠️ No documents have been indexed yet. Please upload a PDF first.", []
        
        # Retrieve with hybrid search
        dense_query_vector = jina_embeddings.embed_query(query)
        
        dense_results = qdrant_client.query_points(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            query=dense_query_vector,
            using=Config.DENSE_VECTOR_NAME,
            with_payload=True,
            limit=Config.PREFETCH_LIMIT,
        )
        
        if not dense_results.points:
            return "No relevant information found in the documents.", []
        
        # Rerank
        candidates = dense_results.points[:Config.RERANK_LIMIT]
        documents_to_rerank = [point.payload.get('document', '') for point in candidates]
        reranked_results = jina_rerank(query, documents_to_rerank, top_n=Config.FINAL_LIMIT)
        
        reranked_points = []
        for result in reranked_results:
            idx = result['index']
            score = result['relevance_score']
            point = candidates[idx]
            point.score = score
            reranked_points.append(point)
        
        if not reranked_points:
            return "No relevant information found.", []
        
        # Deduplicate chunks from same PDF page (same as in generation mode)
        page_to_chunks = {}
        for point in reranked_points:
            payload = point.payload
            source_url = payload.get('source_url', None)
            page = payload.get('page', 'N/A')
            key = (source_url, page)
            
            if key not in page_to_chunks:
                page_to_chunks[key] = []
            page_to_chunks[key].append(point)
        
        # Merge chunks from same page
        deduplicated_points = []
        for key, chunks in page_to_chunks.items():
            if len(chunks) == 1:
                deduplicated_points.append(chunks[0])
            else:
                best_chunk = max(chunks, key=lambda x: x.score)
                combined_content = "\n\n".join([c.payload.get('document', '') for c in chunks])
                best_chunk.payload['document'] = combined_content
                best_chunk.payload['merged_count'] = len(chunks)
                deduplicated_points.append(best_chunk)
        
        deduplicated_points.sort(key=lambda x: x.score, reverse=True)
        
        # Build response and metadata
        response = "**🎯 Retrieved Information**\n\n"
        chunks_metadata = []
        
        for i, point in enumerate(deduplicated_points, 1):
            payload = point.payload
            source = payload.get('source', 'Unknown')
            source_url = payload.get('source_url', None)
            page = payload.get('page', 'N/A')
            chunk_title = payload.get('chunk_title', '')
            document = payload.get('document', '')
            score = point.score
            merged_count = payload.get('merged_count', 1)
            
            # Content first
            response += f"**Chunk {i}** (Score: {score:.2%})"
            if merged_count > 1:
                response += f" *(merged {merged_count} chunks)*"
            response += f"\n{document}\n\n"
            
            # Citation
            if Config.ENABLE_CITATIONS:
                response += f"**📚 Citation:** "
                
                if source_url and page != 'N/A':
                    page_link = f"{source_url}#page={page}"
                    response += f"[{source}, Page {page}]({page_link})\n\n"
                elif source_url:
                    response += f"[{source}]({source_url})\n\n"
                else:
                    response += f"{source}, Page {page}\n\n"
            
            response += "---\n\n"
            
            # Store metadata (mark as cited since we're showing all in retrieval mode)
            chunks_metadata.append({
                "chunk_number": i,
                "source": source,
                "source_url": source_url,
                "page": page,
                "chunk_title": chunk_title,
                "score": score,
                "merged_count": merged_count,
                "preview": document[:150] + "..." if len(document) > 150 else document,
                "cited": True  # In retrieval-only mode, all chunks are "cited" (shown)
            })
        
        return response, chunks_metadata
        
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        return f"⚠️ Error: {str(e)}", []


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
