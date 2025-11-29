## 📄 Document RAG System with AI Generation

A production-grade RAG system with **hybrid search + AI generation** powered entirely by cloud APIs:
- ☁️ **Dense embeddings** via Jina Cloud API (semantic search)
- ☁️ **Sparse BM25** via Qdrant built-in (keyword matching)
- ☁️ **Reranking** via Jina Reranker Cloud API (precision)
- 🤖 **AI Generation** via Google Gemini (natural answers)
- 📚 **Grounded Citations** with clickable links & text highlighting

**Perfect for slow compute** - all processing done in the cloud!

Powered by Qdrant + Google Gemini.

## 🌟 Key Features

- **🤖 AI-Powered Answers**: Google Gemini generates natural responses from retrieved context
- **🎯 Hybrid Search Pipeline**: 
  - ☁️ Dense Embeddings (Jina) + BM25 (Qdrant) → Rerank (Jina) → Generate (Gemini)
- **📚 Grounded Citations**: 
  - Clickable links to source PDFs
  - Text fragment highlighting (jumps to exact text in PDF!)
  - Page-level deep linking
- **⚙️ Dual Modes**: 
  - AI Generation (natural answers with citations)
  - Retrieval Only (raw chunks for debugging)
- **📤 Document Upload**: Upload PDFs via UI or batch process
- **🗄️ Qdrant Vector Store**: Production-ready with web dashboard
- **📊 Structure-Aware Chunking**: Preserves semantic boundaries
- **💾 Persistent Storage**: Data survives restarts
- **⚡ All-Cloud Processing**: Fast even on slow machines
- **🎨 Visual Dashboard**: http://localhost:6333/dashboard

## 🔬 How All-Cloud Hybrid Search Works

```
User Query
    ├── Dense Embeddings (Jina Cloud) ─────┐
    ├── BM25 Text Search (Qdrant)      ────┤──> Fetch 20 results each
    └── Jina Reranker (Cloud)          ────┘──> Rerank → Top 5 Results
```

### Retrieval Pipeline (All Cloud-Based)

1. **Query Embedding**: Query sent to Jina Cloud API
   - ☁️ Dense embedding via Jina Cloud (semantic meaning)
2. **Parallel Search** (Hybrid):
   - Dense semantic search via Jina embeddings
   - BM25 keyword search via Qdrant's built-in text indexing
3. **Fusion**: Results are combined and deduplicated
4. **Reranking**: Jina Reranker API reranks for final precision
5. **Final Results**: Top-k most relevant chunks

**All processing in the cloud = works great on slow machines!**

### Why All-Cloud Hybrid?

| Method | Strengths | Weaknesses | Source |
|--------|-----------|------------|--------|
| **Dense** | Semantic meaning, synonyms | Misses exact keywords | ☁️ Jina Cloud |
| **BM25** | Exact keyword matching | No semantic understanding | ☁️ Qdrant Built-in |
| **Reranking** | Precision scoring | Needs candidates first | ☁️ Jina Cloud |
| **Hybrid** | ✅ Best of all worlds | Requires API key | ☁️ All Cloud |

**Perfect for slow machines** - no local model loading or inference!

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This includes:
- `langchain-community` - Jina Cloud API integration
- `qdrant-client` - Vector database client
- `requests` - For Jina Reranker API
- `streamlit` - Web interface
- And more...

**No FastEmbed or local models** - everything runs in the cloud!

### 2. Setup Environment

Copy the example environment file:

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```env
JINA_API_KEY=your_jina_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=hybrid-search
```

Get your API keys:
- **Jina**: [jina.ai](https://jina.ai) (free tier: 1M tokens/month)
- **Google**: [makersuite.google.com](https://makersuite.google.com/app/apikey) (free tier: 60 queries/min)

### 3. Start Qdrant

```bash
docker-compose up -d
```

Verify Qdrant is running:

```bash
curl http://localhost:6333/health
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will:
- Connect to Jina Cloud (embeddings + reranking)
- Connect to Google Gemini (AI generation)
- Use Qdrant's built-in BM25 for keyword search
- Auto-process PDFs in `data/` folder
- Open at http://localhost:8501

**Note**: First run is instant - no model downloads needed! All processing is cloud-based.

### Usage Modes

**🤖 AI Generation Mode** (Default):
- Natural language answers
- Synthesizes information from multiple sources
- Includes grounded citations with links

**📄 Retrieval Only Mode**:
- Shows raw retrieved chunks
- Good for debugging/verification
- See exactly what the AI sees

Toggle between modes in the sidebar!

## 📊 Access Points

- **Streamlit App**: http://localhost:8501
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Qdrant API**: http://localhost:6333

## 📚 Usage

### Upload Documents

**Option A**: Streamlit UI
- Use the file uploader
- Automatic processing with all three embedding types

**Option B**: Batch Processing
- Place PDFs in `data/` folder
- Restart app to auto-process

### Search Documents

Type your query in the chat interface. The system:
1. Generates query embeddings (all three types)
2. Runs hybrid search (prefetch + rerank)
3. Returns top results with scores

Example queries:
- "What are the fee payment deadlines?"
- "scholarship eligibility criteria"
- "How to apply for financial aid?"

### Explore Results

Each result shows:
- **Score**: Relevance after reranking
- **Source**: File name
- **Page**: Page number
- **Chunk Title**: Section context
- **Content**: Actual text

## 🏗️ Project Structure

```
├── docker-compose.yml        # Qdrant container
├── config.py                 # Hybrid search configuration
├── vectorstore.py            # Multi-embedding ingestion
├── rag_chain.py              # Hybrid retrieval with prefetch/rerank
├── app.py                    # Streamlit interface
├── preprocessing_simple.py   # PDF chunking
├── clear_index.py            # Reset utility
├── requirements.txt          # Python dependencies
├── env.example               # Environment template
├── .streamlit/
│   └── config.toml           # Streamlit settings
├── data/                     # PDF files
├── qdrant_data/              # Vector storage (Docker volume)
└── qdrant_metadata/          # Processing metadata
```

## ⚙️ Configuration

Edit `config.py` to customize:

### Embedding Models

```python
# Dense Embeddings (Jina Cloud API)
JINA_DENSE_MODEL = "jina-embeddings-v2-base-en"
DENSE_DIMENSION = 768

# Sparse Embeddings (FastEmbed Local - BM25)
SPARSE_MODEL = "Qdrant/bm25"

# Late Interaction (FastEmbed Local - ColBERT)
LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"
LATE_INTERACTION_DIMENSION = 128
```

### Retrieval Parameters

```python
PREFETCH_LIMIT = 20  # Results from each sub-query
FINAL_LIMIT = 5      # Final results after reranking
```

### Chunking

```python
CHUNKING_STRATEGY = "semantic"  # or "fixed"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
```

## 🔧 Advanced Usage

### Test Individual Retrieval Methods

In Python:

```python
from rag_chain import retrieve_dense_only, retrieve_sparse_only, chat_with_document

# Dense only (semantic)
results = retrieve_dense_only("your query", k=5)

# Sparse only (BM25)
results = retrieve_sparse_only("your query", k=5)

# Hybrid (recommended)
results = chat_with_document("your query")
```

### Adjust Retrieval Parameters

```python
from rag_chain import retrieve_relevant_chunks_hybrid

# Custom prefetch and final limits
results = retrieve_relevant_chunks_hybrid(
    query="your query",
    prefetch_limit=30,  # More candidates
    final_limit=10      # More final results
)
```

### Inspect Collection

```python
from qdrant_client import QdrantClient
from config import Config

client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)

# Get collection info
info = client.get_collection(Config.QDRANT_COLLECTION_NAME)
print(f"Points: {info.points_count}")
print(f"Vectors: {info.config.params.vectors}")
```

## 🎯 Performance Tuning

### For Better Recall

Increase prefetch limits in `config.py`:

```python
PREFETCH_LIMIT = 30  # More candidates for reranking
FINAL_LIMIT = 10     # More final results
```

### For Better Precision

Keep prefetch limits lower and rely on reranking:

```python
PREFETCH_LIMIT = 15
FINAL_LIMIT = 3
```

### For Speed

- Reduce chunk size: `CHUNK_SIZE = 500`
- Lower prefetch limit: `PREFETCH_LIMIT = 10`
- Disable late interaction (not recommended)

## 📈 Understanding the Models

### Dense Embeddings (all-MiniLM-L6-v2)
- **Size**: 384 dimensions
- **Best for**: Semantic similarity, paraphrasing
- **Example**: "price" matches "cost", "fee", "payment"

### Sparse Embeddings (BM25)
- **Type**: Term frequency-based
- **Best for**: Exact keyword matching
- **Example**: "scholarship" matches documents with that exact term

### Late Interaction (ColBERT)
- **Size**: 128 dimensions (multi-vector)
- **Best for**: Token-level interactions, reranking
- **How**: Compares query tokens with document tokens

## 🐛 Troubleshooting

### "Failed to connect to Qdrant"

```bash
# Check if running
docker-compose ps

# Check health
curl http://localhost:6333/health

# Start if not running
docker-compose up -d
```

### "No documents indexed"

1. Place PDFs in `data/` folder
2. Restart Streamlit app
3. Check logs for errors

### Slow Performance

**Model Loading**: First run loads models (1-2 minutes)
- Dense model: ~30MB
- BM25 model: Lightweight
- ColBERT model: ~110MB

**Solution**: Models are cached after first load

### Out of Memory

Reduce batch size or chunk size:

```python
CHUNK_SIZE = 500  # Smaller chunks
CHUNK_OVERLAP = 100
```

### Clear and Restart

```bash
python clear_index.py
docker-compose restart qdrant
streamlit run app.py
```

## 📖 API Examples

### REST API Search

```bash
# Query the collection
curl -X POST http://localhost:6333/collections/hybrid-search/points/query \
  -H "Content-Type: application/json" \
  -d '{
    "prefetch": [
      {
        "query": [0.1, 0.2, ...],
        "using": "all-MiniLM-L6-v2",
        "limit": 20
      }
    ],
    "query": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "using": "colbertv2.0",
    "limit": 5
  }'
```

### Collection Stats

```bash
curl http://localhost:6333/collections/hybrid-search
```

## 🔬 Technical Details

### Collection Configuration

```python
vectors_config={
    "all-MiniLM-L6-v2": {
        "size": 384,
        "distance": "COSINE"
    },
    "colbertv2.0": {
        "size": 128,
        "distance": "COSINE",
        "multivector_config": {
            "comparator": "MAX_SIM"
        },
        "hnsw_config": {"m": 0}  # No indexing for reranking
    }
},
sparse_vectors_config={
    "bm25": {
        "modifier": "IDF"
    }
}
```

### Why These Models?

1. **Jina v2 base** (Cloud): High-quality, 768-dim semantic embeddings from Jina AI
2. **Qdrant/bm25** (Local): Optimized sparse vectors for keyword search
3. **ColBERT v2** (Local): State-of-the-art late interaction model

**Best of both worlds**: Cloud quality for dense + Local privacy for sparse/reranking

### Prefetch Strategy

Prefetch runs multiple searches in parallel:
- Each search returns top-N results
- Results are fused
- Reranking refines the final order

This is more effective than:
- Single embedding type
- Simple concatenation
- Post-hoc fusion

## 📚 Resources

- [Qdrant Hybrid Search Guide](https://qdrant.tech/documentation/guides/hybrid-search/)
- [FastEmbed Documentation](https://qdrant.github.io/fastembed/)
- [ColBERT Paper](https://arxiv.org/abs/2004.12832)
- [Qdrant Dashboard](http://localhost:6333/dashboard)

## 🎓 Common Commands

```bash
# Start everything
docker-compose up -d
streamlit run app.py

# Check Qdrant
curl http://localhost:6333/health

# Clear and restart
python clear_index.py
docker-compose restart qdrant

# View logs
docker-compose logs -f qdrant

# Stop everything
docker-compose down
```

## 💡 Best Practices

1. **Chunking**: Use semantic chunking for structured documents
2. **Prefetch Limit**: Start with 20, adjust based on results
3. **Final Limit**: 3-5 for focused results, 10+ for broad coverage
4. **Model Loading**: First run is slow (model download), then fast
5. **Experimentation**: Use Qdrant dashboard to visualize embeddings

## 🚀 Next Steps

1. **Upload Documents**: Add PDFs to `data/` folder
2. **Test Search**: Try various query types
3. **Explore Dashboard**: Visualize embeddings
4. **Tune Parameters**: Adjust prefetch/final limits in `config.py`
5. **Compare Methods**: Test dense-only vs sparse-only vs hybrid

## 📄 License

MIT

---

**Built with** FastEmbed • Qdrant • Streamlit • LangChain
