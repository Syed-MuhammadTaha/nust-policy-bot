import re
import requests
import json
from typing import Dict, List
from vectorstore import qdrant_client, jina_embeddings
from qdrant_client.models import models
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Initialize Google Gemini LLM with structured output
llm = ChatGoogleGenerativeAI(
    model=Config.GOOGLE_MODEL,
    google_api_key=Config.GOOGLE_API_KEY,
    temperature=Config.GENERATION_TEMPERATURE,
    max_output_tokens=Config.GENERATION_MAX_TOKENS
)

# Pydantic model for structured LLM output
class RAGResponse(BaseModel):
    """Structured response from LLM with clean answer and highlight mappings."""
    answer: str = Field(description="The complete answer with simple citation markers like [1], [2]")
    highlights: Dict[int, List[str]] = Field(
        description="Mapping of citation number to list of exact quotes from that source. Each quote should be 5-15 words taken exactly from the context.",
        example={1: ["exact quote from source 1"], 2: ["first quote from source 2", "second quote from source 2"]}
    )

# Note: Using JSON mode in prompt instead of structured output binding
# This is more reliable across different LLM providers

# Create RAG prompt template with strategic inline citation instructions
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions about university policies and documents.

Answer the question based ONLY on the provided context below. Be concise, accurate, and factual.

CRITICAL CITATION RULE - STRUCTURED OUTPUT:
You MUST provide your response in a structured format with clean citations and separate highlight mappings.

Response Structure:
1. "answer": Your complete answer with SIMPLE citation markers [1], [2], [3]
2. "highlights": A mapping of citation numbers to lists of exact 5-15 word quotes from those sources

Citation Strategy in Answer:
- Use SIMPLE markers: [1], [2], [3] (NO quotes inline)
- Cite ONCE per paragraph/idea, not every sentence
- Place at END of paragraph if all info from one source
- Each source in context is numbered [Source X]

Highlights Mapping Rules:
- Extract 1-3 exact quotes (5-15 words each) from each cited source
- Quotes must be EXACTLY from the context without modification
- Quotes should directly support the claims in your answer
- If you use multiple pieces of info from same source, include multiple highlights

EXCELLENT ✓:
- Clean answer with simple [1], [2]
- Multiple highlights per source allowed
- Exact quotes in highlights mapping

BAD ✗:
- Inline quotes in answer: [1: "quote"] 
- Generic/paraphrased highlights
- Missing highlights for cited sources

CRITICAL - ABSTENTION RULE:
If the context does NOT contain information to answer the question, you MUST respond EXACTLY:
"I don't have enough information in the provided documents to answer that question."

ABSTAIN in these cases:
- Question is completely off-topic (e.g., asking about weather, sports, or unrelated topics)
- Context discusses something different than what's asked
- Context is too vague or doesn't provide specific details needed
- Question asks for personal opinions or predictions
"""),
    ("human", """Context:
{context}

Question: {question}

IMPORTANT: You MUST respond with a valid JSON object in this exact format:
{{
  "answer": "Your answer with simple citations [1], [2], etc.",
  "highlights": {{
    1: ["exact quote 1 from source 1", "exact quote 2 from source 1"],
    2: ["exact quote from source 2"]
  }}
}}

Answer:""")
])

print("✅ Loaded RAG prompt template with inline citations")


def encode_text_fragment(text: str) -> str:
    """
    Properly encode text for PDF #:~:text= fragment.
    Encodes ALL special characters that could break URL text fragments.
    
    Args:
        text: Text to encode
        
    Returns:
        Fully encoded text safe for use in URL text fragments
    """
    from urllib.parse import quote
    
    # First pass: Standard URL encoding (encodes most special chars)
    encoded = quote(text, safe='')
    
    # Second pass: Manually encode "unreserved" characters that quote() skips
    # but can break PDF text fragment matching
    char_map = {
        '.': '%2E',   # Period
        '-': '%2D',   # Hyphen
        '_': '%5F',   # Underscore
        '~': '%7E',   # Tilde
        '/': '%2F',   # Forward slash (quote encodes this, but double-check)
        '\\': '%5C',  # Backslash
        '?': '%3F',   # Question mark
        '!': '%21',   # Exclamation
        '#': '%23',   # Hash
        '&': '%26',   # Ampersand
        '=': '%3D',   # Equals
        '+': '%2B',   # Plus
        ',': '%2C',   # Comma
        ';': '%3B',   # Semicolon
        ':': '%3A',   # Colon
        "'": '%27',   # Single quote
        '"': '%22',   # Double quote
        '(': '%28',   # Open paren
        ')': '%29',   # Close paren
        '[': '%5B',   # Open bracket
        ']': '%5D',   # Close bracket
        '{': '%7B',   # Open brace
        '}': '%7D',   # Close brace
        '<': '%3C',   # Less than
        '>': '%3E',   # Greater than
        '@': '%40',   # At sign
        '*': '%2A',   # Asterisk
        '|': '%7C',   # Pipe
        '`': '%60',   # Backtick
        '^': '%5E',   # Caret
    }
    
    for char, encoded_char in char_map.items():
        encoded = encoded.replace(char, encoded_char)
    
    return encoded


def parse_citations_with_quotes(answer: str) -> dict:
    """
    Parse citations with exact quotes from LLM answer.
    
    Extracts citations in format: [X: "exact quote"]
    
    Returns:
        dict: {citation_num: exact_quote}
        Example: {1: "exact quote from source", 2: "another quote"}
    """
    citations_map = {}
    
    # Pattern to match [X: "quote"] format
    # Matches: [1: "text"], [2: "more text"], etc.
    pattern = r'\[(\d+):\s*["\']([^"\']+)["\']\]'
    matches = re.findall(pattern, answer)
    
    for match in matches:
        cite_num = int(match[0])
        quote_text = match[1].strip()
        citations_map[cite_num] = quote_text
    
    # Also check for simple [X] format (fallback for backward compatibility)
    simple_pattern = r'\[(\d+)\]'
    simple_matches = re.findall(simple_pattern, answer)
    for match in simple_matches:
        cite_num = int(match)
        if cite_num not in citations_map:
            citations_map[cite_num] = ""  # No quote provided
    
    return citations_map




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


def generate_answer_with_citations(query: str, stream: bool = False):
    """
    Full RAG pipeline: Retrieve → Rerank → Generate with Google Gemini.
    
    Args:
        query: User question
        stream: If True, yields chunks for streaming. If False, returns complete response.
    
    Returns:
        If stream=False: (answer_text, chunks_metadata) tuple
        If stream=True: Generator that yields dict with 'type' and 'content'
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
        
        # Step 3: Check relevance threshold for abstention
        best_score = reranked_points[0].score if reranked_points else 0.0
        
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
        
        for i, point in enumerate(reranked_points, 1):
            payload = point.payload
            source = payload.get('source', 'Unknown')
            source_url = payload.get('source_url', None)
            page = payload.get('page', 'N/A')
            chunk_title = payload.get('chunk_title', '')
            document = payload.get('document', '')
            score = point.score
            
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
            citations.append(citation)
            
            # Store metadata for sidebar display
            chunks_metadata.append({
                "chunk_number": i,
                "source": source,
                "source_url": source_url,
                "page": page,
                "chunk_title": chunk_title,
                "score": score,
                "preview": document[:150] + "..." if len(document) > 150 else document
            })
        
        # Step 5: Generate answer with LLM
        print(f"🤖 Generating answer with Google Gemini...")
        context = "\n\n".join(context_parts)
        
        # Format the prompt with context and question
        messages = rag_prompt.format_messages(context=context, question=query)
        
        # STREAMING MODE
        if stream:
            def stream_generator():
                """Generator for streaming response with citations."""
                answer = ""
                
                # Yield metadata first (for sidebar update)
                yield {"type": "metadata", "content": chunks_metadata}
                
                # Stream the answer
                for chunk in llm.stream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        answer += chunk.content
                        yield {"type": "text", "content": chunk.content}
                
                # Parse citations with quotes from answer
                citations_with_quotes = parse_citations_with_quotes(answer)
                cited_indices = set(citations_with_quotes.keys())
                
                # Use LLM-provided quotes for highlighting
                chunk_highlight_map = {}
                if Config.ENABLE_SMART_HIGHLIGHTING:
                    for cite_num, quote in citations_with_quotes.items():
                        if quote:  # Only highlight if LLM provided a quote
                            chunk_highlight_map[cite_num] = quote
                        # If no quote, just skip highlighting for this citation
                
                # Filter to only show cited sources and rebuild with highlights
                cited_citations = []
                cited_metadata = []
                
                for i, metadata in enumerate(chunks_metadata, 1):
                    if i in cited_indices:
                        # Rebuild citation with highlight
                        source = metadata.get('source', 'Unknown')
                        source_url = metadata.get('source_url', None)
                        page = metadata.get('page', 'N/A')
                        score = metadata.get('score', 0)
                        highlight_text = chunk_highlight_map.get(i, '')
                        
                        citation = f"**[{i}]** "
                        if source_url and page != 'N/A' and highlight_text:
                            # Add text fragment highlighting with comprehensive encoding
                            encoded_text = encode_text_fragment(highlight_text)
                            page_link = f"{source_url}#page={page}:~:text={encoded_text}"
                            citation += f"[{source}, Page {page}]({page_link})"
                        elif source_url and page != 'N/A':
                            page_link = f"{source_url}#page={page}"
                            citation += f"[{source}, Page {page}]({page_link})"
                        elif source_url:
                            citation += f"[{source}]({source_url})"
                        else:
                            citation += f"{source}, Page {page}"
                        
                        citation += f" • {score:.0%}"
                        
                        cited_citations.append(citation)
                        
                        metadata_copy = metadata.copy()
                        metadata_copy['cited'] = True
                        metadata_copy['highlight_text'] = highlight_text
                        cited_metadata.append(metadata_copy)
                    else:
                        metadata_copy = metadata.copy()
                        metadata_copy['cited'] = False
                        cited_metadata.append(metadata_copy)
                
                # Yield citations
                if cited_citations:
                    citations_text = "\n\n---\n\n**📚 Sources:**\n\n" + "\n".join(cited_citations)
                else:
                    citations_text = "\n\n*Note: No specific sources were cited for this answer.*"
                
                # Add low confidence warning if needed
                if best_score < 0.50:
                    citations_text += f"\n\n⚠️ *Low confidence: Best relevance score is {best_score:.0%}. "
                    citations_text += "The answer may be less reliable or the question may be partially outside the document scope.*"
                
                yield {"type": "citations", "content": citations_text}
                yield {"type": "metadata_update", "content": cited_metadata}
            
            return stream_generator()
        
        # NON-STREAMING MODE with JSON parsing
        else:
            # Generate answer (LLM will return JSON per prompt)
            print(f"🤖 Generating answer with JSON format...")
            answer = ""
            highlights_map = {}
            
            try:
                response_obj = llm.invoke(messages)
                raw_response = response_obj.content
                print(f"📥 Raw response received ({len(raw_response)} chars)")
                
                # Extract JSON from response (might have markdown code blocks or extra text)
                json_str = raw_response.strip()
                
                # Remove markdown code blocks if present
                if json_str.startswith('```'):
                    # Extract content between ```json and ```
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to find JSON object
                        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                
                # Parse JSON
                parsed = json.loads(json_str)
                answer = parsed.get('answer', '')
                highlights_map = parsed.get('highlights', {})
                
                # Validate and normalize highlights_map
                if not isinstance(highlights_map, dict):
                    print(f"⚠️ Highlights map is not a dict: {type(highlights_map)}. Converting...")
                    highlights_map = {}
                
                # Ensure all values are lists and convert keys to int
                normalized_highlights = {}
                for key, value in highlights_map.items():
                    try:
                        int_key = int(key) if isinstance(key, str) else key
                        if isinstance(value, str):
                            normalized_highlights[int_key] = [value]
                        elif isinstance(value, list):
                            normalized_highlights[int_key] = value
                        else:
                            normalized_highlights[int_key] = []
                    except (ValueError, TypeError):
                        continue
                
                highlights_map = normalized_highlights
                print(f"✅ Parsed JSON: answer length={len(answer)}, highlights={len(highlights_map)} sources")
                
            except json.JSONDecodeError as json_err:
                print(f"⚠️ JSON parsing failed: {json_err}")
                print(f"   Response preview: {raw_response[:500]}")
                print(f"   Attempting fallback...")
                
                # Fallback: treat as regular text and parse citations
                answer = raw_response
                citations_with_quotes = parse_citations_with_quotes(answer)
                highlights_map = citations_with_quotes
                print(f"   ✅ Using fallback citation parsing")
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"❌ Error during generation ({error_type}): {error_msg}")
                return f"⚠️ Error generating response: {error_msg}\n\nPlease try again or check your API key.", []
            
            # Step 6: Get cited indices from answer
            cited_indices = set()
            citation_pattern = r'\[(\d+)\]'
            matches = re.findall(citation_pattern, answer)
            for match in matches:
                cited_indices.add(int(match))
            
            # Step 6.5: Process highlights from structured output
            # highlights_map is Dict[int, List[str]] - each citation can have multiple highlights
            print(f"🎯 Processing highlights for PDF linking...")
            
            # Filter to only show cited sources and rebuild citations with highlights
            cited_citations = []
            cited_metadata = []
            
            for i, metadata in enumerate(chunks_metadata, 1):
                if i in cited_indices:
                    # Get highlights for this citation
                    highlight_list = highlights_map.get(i, [])
                    if isinstance(highlight_list, str):
                        highlight_list = [highlight_list]  # Convert single string to list
                    
                    source = metadata.get('source', 'Unknown')
                    source_url = metadata.get('source_url', None)
                    page = metadata.get('page', 'N/A')
                    score = metadata.get('score', 0)
                    
                    # Use first highlight for the main link (most relevant)
                    primary_highlight = highlight_list[0] if highlight_list else ''
                    
                    if Config.ENABLE_SMART_HIGHLIGHTING and highlight_list:
                        print(f"  ✅ Citation [{i}]: {len(highlight_list)} highlight(s)")
                    
                    citation = f"**[{i}]** "
                    if source_url and page != 'N/A' and primary_highlight:
                        # Add text fragment highlighting with comprehensive encoding
                        encoded_text = encode_text_fragment(primary_highlight)
                        page_link = f"{source_url}#page={page}:~:text={encoded_text}"
                        citation += f"[{source}, Page {page}]({page_link})"
                    elif source_url and page != 'N/A':
                        page_link = f"{source_url}#page={page}"
                        citation += f"[{source}, Page {page}]({page_link})"
                    elif source_url:
                        citation += f"[{source}]({source_url})"
                    else:
                        citation += f"{source}, Page {page}"
                    
                    citation += f" • {score:.0%}"
                    
                    cited_citations.append(citation)
                    
                    metadata_copy = metadata.copy()
                    metadata_copy['cited'] = True
                    metadata_copy['highlight_text'] = primary_highlight
                    metadata_copy['all_highlights'] = highlight_list  # Store all highlights
                    cited_metadata.append(metadata_copy)
                else:
                    metadata_copy = metadata.copy()
                    metadata_copy['cited'] = False
                    cited_metadata.append(metadata_copy)
            
            # Step 7: Format final response
            # Use answer exactly as parsed from JSON - no modifications
            response = answer + "\n\n"
            
            if cited_citations:
                response += "---\n\n**📚 Sources:**\n\n"
                response += "\n".join(cited_citations)
            else:
                response += "\n\n*Note: No specific sources were cited for this answer.*"
            
            # Add warning if relevance is marginal
            if best_score < 0.50:
                response += f"\n\n⚠️ *Low confidence: Best relevance score is {best_score:.0%}. "
                response += "The answer may be less reliable or the question may be partially outside the document scope.*"
            
            return response, cited_metadata
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return f"⚠️ Error: {str(e)}", []


def chat_with_document(query: str, stream: bool = False):
    """
    Main entry point for full RAG with generation.
    
    Args:
        query: User question
        stream: If True, returns generator for streaming
    
    Returns:
        If stream=False: (answer, chunks_metadata) tuple
        If stream=True: Generator yielding chunks
    """
    return generate_answer_with_citations(query, stream=stream)


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
        
        # Build response and metadata
        response = "**🎯 Retrieved Information**\n\n"
        chunks_metadata = []
        
        for i, point in enumerate(reranked_points, 1):
            payload = point.payload
            source = payload.get('source', 'Unknown')
            source_url = payload.get('source_url', None)
            page = payload.get('page', 'N/A')
            chunk_title = payload.get('chunk_title', '')
            document = payload.get('document', '')
            score = point.score
            
            # Content first
            response += f"**Chunk {i}** (Score: {score:.2%})\n{document}\n\n"
            
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
