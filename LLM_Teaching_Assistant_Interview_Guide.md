# 🎯 LLM Teaching Assistant - Complete Interview Prep Guide

> **Your Goal**: Be able to explain every concept in your project at surface level (BFS) AND deep technical level (DFS)

---

# 📚 Table of Contents

1. [Project Overview](#1-project-overview)
2. [RAG (Retrieval-Augmented Generation)](#2-rag-retrieval-augmented-generation)
3. [Vector Embeddings & Similarity Search](#3-vector-embeddings--similarity-search)
4. [FAISS - Vector Database](#4-faiss---vector-database)
5. [LLMs & Prompt Engineering](#5-llms--prompt-engineering)
6. [System Architecture](#6-system-architecture)
7. [Backend Deep Dive](#7-backend-deep-dive)
8. [Frontend Deep Dive](#8-frontend-deep-dive)
9. [DevOps & Deployment](#9-devops--deployment)
10. [Common Interview Questions](#10-common-interview-questions)

---

# 1. Project Overview

## BFS (High-Level Explanation)
"I built an AI-powered learning platform that transforms complex research papers into beginner-friendly lessons. Users type a question like 'Explain attention mechanisms', and the system finds the most relevant paper, parses it, and generates educational content section by section."

## DFS (Technical Deep-Dive)

### What Problem Does It Solve?
```
Traditional Approach:
User → ChatGPT → Generic answer (may hallucinate, no sources)

My Approach:
User → Semantic Search → Find Real Paper → Parse PDF → Generate Grounded Lessons
```

### Technical Flow
```
1. User Query: "Explain transformers"
                    ↓
2. Embed Query: OpenAI text-embedding-3-small → 1536-dim vector
                    ↓
3. FAISS Search: Find nearest neighbor from 231 indexed papers
                    ↓
4. Fetch Paper: Download PDF from arXiv
                    ↓
5. Parse PDF: GROBID extracts sections (intro, methods, results...)
                    ↓
6. Generate Lessons: GPT-4o-mini creates beginner-friendly content per section
                    ↓
7. Return: Structured lesson with citations
```

### Why This Architecture?
| Decision | Why |
|----------|-----|
| RAG over fine-tuning | Cheaper, updatable, no training needed |
| FAISS over Pinecone | Free, local, fast for small datasets |
| GPT-4o-mini over GPT-4 | 10x cheaper, sufficient quality for lessons |
| GROBID over regex | Handles complex PDFs, extracts structure |
| FastAPI over Flask | Async, faster, auto-docs, type hints |

---

# 2. RAG (Retrieval-Augmented Generation)

## BFS (Simple Explanation)
"RAG combines the best of search engines and language models. Instead of asking an LLM to remember everything, we first RETRIEVE relevant documents, then AUGMENT the prompt with that context, and finally GENERATE an answer grounded in real sources."

## DFS (Technical Deep-Dive)

### Why RAG Exists
```
Problem with Pure LLMs:
- Training data has a cutoff date
- Can hallucinate facts
- Can't cite sources
- Expensive to update (requires retraining)

RAG Solution:
- Retrieves current information
- Grounds responses in real documents
- Can cite exact sources
- Update by adding new documents (no retraining)
```

### RAG Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Query   │───▶│   Retriever  │───▶│    Generator     │   │
│  │          │    │   (Search)   │    │     (LLM)        │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│                         │                     │              │
│                         ▼                     ▼              │
│                  ┌─────────────┐      ┌─────────────┐       │
│                  │  Document   │      │  Grounded   │       │
│                  │    Store    │      │   Answer    │       │
│                  └─────────────┘      └─────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### RAG Components in My Project

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| **Document Store** | FAISS index + urls.json | Stores 231 paper embeddings |
| **Retriever** | Semantic search with cosine similarity | Finds relevant papers |
| **Generator** | GPT-4o-mini | Creates lessons from retrieved content |
| **Augmentation** | Paper sections injected into prompt | Grounds the generation |

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Cost** | Low (just API calls) | High (training compute) |
| **Update data** | Add documents | Retrain model |
| **Hallucination** | Reduced (grounded) | Still possible |
| **Latency** | Higher (retrieval step) | Lower |
| **Transparency** | Can cite sources | Black box |
| **When to use** | Dynamic knowledge, need citations | Static domain, need speed |

### Advanced RAG Techniques (Know These!)

```
Basic RAG (What I Built):
Query → Single retrieval → Generate

Advanced RAG:
1. Query Rewriting: LLM reformulates query for better retrieval
2. Hybrid Search: Combine semantic + keyword search
3. Re-ranking: Score retrieved docs with cross-encoder
4. Multi-hop: Retrieve → Generate partial → Retrieve more → Generate final
5. Self-RAG: Model decides when to retrieve
```

### Code Example from My Project
```python
# From teaching_service.py
async def teach(self, query: str, ...) -> Lesson:
    # 1. RETRIEVE: Find relevant paper
    search_results = self.paper_service.search(query, top_k=1)
    paper = self.paper_service.get_paper(search_results[0].paper.url)
    
    # 2. AUGMENT: Paper content becomes context
    # 3. GENERATE: Create lessons grounded in paper
    lesson = await self.lesson_service.generate_lesson(
        paper=paper,
        query=query,
        ...
    )
    return lesson
```

---

# 3. Vector Embeddings & Similarity Search

## BFS (Simple Explanation)
"Embeddings convert text into numbers (vectors) that capture meaning. Similar texts have similar vectors. We use this to find papers that match a user's question, even if they don't share exact words."

## DFS (Technical Deep-Dive)

### What Are Embeddings?
```
Text: "The cat sat on the mat"
         ↓ Embedding Model
Vector: [0.023, -0.156, 0.892, ..., 0.445]  # 1536 dimensions

Key Insight: Similar meanings → Similar vectors
- "The cat sat on the mat" ≈ "A feline rested on the rug"
- "The cat sat on the mat" ≠ "Stock prices rose today"
```

### Why 1536 Dimensions?
- More dimensions = more semantic nuance captured
- OpenAI's text-embedding-3-small uses 1536
- Each dimension represents some learned "feature" of meaning
- Trade-off: More dimensions = better quality but more storage/compute

### Embedding Models Comparison

| Model | Dimensions | Quality | Speed | Cost |
|-------|------------|---------|-------|------|
| text-embedding-3-small | 1536 | Good | Fast | $0.02/1M tokens |
| text-embedding-3-large | 3072 | Better | Slower | $0.13/1M tokens |
| text-embedding-ada-002 | 1536 | Good | Fast | $0.10/1M tokens |
| BERT (local) | 768 | Decent | Fast | Free |
| Sentence-BERT (local) | 384-768 | Good | Fast | Free |

### Similarity Metrics

#### Cosine Similarity (What I Use)
```
Formula: cos(θ) = (A · B) / (||A|| × ||B||)

Range: -1 to 1
- 1 = identical direction (same meaning)
- 0 = perpendicular (unrelated)
- -1 = opposite direction

Why Cosine?
- Ignores magnitude, only cares about direction
- Works well for normalized embeddings
- Most common for text similarity
```

```python
# Implementation
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example
query_vec = [0.1, 0.2, 0.3]
doc_vec = [0.15, 0.25, 0.28]
similarity = cosine_similarity(query_vec, doc_vec)  # ~0.99 (very similar)
```

#### Other Similarity Metrics
```
Euclidean Distance: sqrt(Σ(a_i - b_i)²)
- Measures absolute distance
- Affected by magnitude
- Lower = more similar

Dot Product: Σ(a_i × b_i)
- Affected by magnitude
- Fast to compute
- Used when vectors are normalized

Manhattan Distance: Σ|a_i - b_i|
- Sum of absolute differences
- Less sensitive to outliers
```

### How Embeddings Are Created (Transformer Architecture)

```
Input: "What is attention?"
           ↓
┌─────────────────────────────────┐
│      TOKENIZATION               │
│  ["What", "is", "attention", "?"]│
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│    TOKEN EMBEDDINGS             │
│  Each token → initial vector    │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  TRANSFORMER LAYERS (12-24)     │
│  Self-attention + Feed-forward  │
│  Tokens "see" each other        │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│      POOLING                    │
│  Combine all tokens → 1 vector  │
│  (mean pooling or [CLS] token)  │
└─────────────────────────────────┘
           ↓
Output: [0.023, -0.156, ..., 0.445]  # 1536-dim
```

### Code from My Project
```python
# From embedding_service.py
class EmbeddingService:
    def __init__(self):
        self.client = OpenAI()
        self.model = "text-embedding-3-small"
    
    def embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return np.array([d.embedding for d in response.data], dtype=np.float32)
```

---

# 4. FAISS - Vector Database

## BFS (Simple Explanation)
"FAISS is Facebook's library for fast similarity search. It stores vectors and quickly finds the most similar ones to a query. Think of it as a smart index that can search millions of vectors in milliseconds."

## DFS (Technical Deep-Dive)

### Why FAISS?
```
Naive Search: Compare query to ALL vectors → O(n) 
- 1M vectors × 1536 dims = 1.5B operations per search
- Way too slow!

FAISS: Smart indexing structures → O(log n) or better
- Uses approximations and clever data structures
- Trades tiny accuracy loss for massive speed gains
```

### FAISS Index Types

#### 1. Flat Index (Exact Search) - What I Use
```python
index = faiss.IndexFlatIP(1536)  # Inner Product (cosine for normalized)
index = faiss.IndexFlatL2(1536)  # Euclidean distance
```
- **How it works**: Brute force, compares to every vector
- **Pros**: 100% accurate
- **Cons**: Slow for large datasets
- **Use when**: < 100K vectors (my case: 231 vectors)

#### 2. IVF (Inverted File Index) - For Medium Scale
```python
quantizer = faiss.IndexFlatL2(1536)
index = faiss.IndexIVFFlat(quantizer, 1536, nlist=100)
index.train(vectors)  # Must train!
```
- **How it works**: 
  - Clusters vectors into `nlist` groups
  - At search time, only searches `nprobe` nearest clusters
- **Pros**: Much faster than flat
- **Cons**: Approximate, requires training
- **Use when**: 100K - 1M vectors

```
Visual:
┌─────────────────────────────────────┐
│         Vector Space                │
│   ┌───┐   ┌───┐   ┌───┐   ┌───┐    │
│   │ 1 │   │ 2 │   │ 3 │   │ 4 │    │  ← Clusters
│   │•••│   │•• │   │•••│   │•  │    │
│   │ • │   │•••│   │ • │   │•••│    │
│   └───┘   └───┘   └───┘   └───┘    │
│                                     │
│   Query lands in cluster 2          │
│   → Only search cluster 2 (+ maybe 1,3) │
└─────────────────────────────────────┘
```

#### 3. HNSW (Hierarchical Navigable Small World) - For Speed
```python
index = faiss.IndexHNSWFlat(1536, 32)  # 32 = connections per node
```
- **How it works**: 
  - Builds a graph where similar vectors are connected
  - Search navigates the graph greedily
- **Pros**: Very fast, good recall
- **Cons**: High memory usage, slow to build
- **Use when**: Need fastest search, have memory

```
Visual:
Layer 2:  A ─────────── B          (sparse, long jumps)
          │             │
Layer 1:  A ─── C ─── B ─── D      (medium density)
          │     │     │     │
Layer 0:  A─E─C─F─B─G─D─H─...      (dense, all vectors)

Search: Start at top layer, greedily descend
```

#### 4. PQ (Product Quantization) - For Memory
```python
index = faiss.IndexPQ(1536, 64, 8)  # 64 subvectors, 8 bits each
```
- **How it works**: 
  - Compresses vectors by splitting into subvectors
  - Each subvector quantized to nearest centroid
- **Pros**: 10-100x memory reduction
- **Cons**: Lossy compression, lower accuracy
- **Use when**: Billions of vectors, limited RAM

### My FAISS Implementation
```python
# From embedding_service.py
class EmbeddingService:
    def __init__(self):
        self.index = None
        self.urls = []
    
    def build_index(self, embeddings: np.ndarray, urls: list[str]):
        """Build FAISS index from embeddings."""
        dim = embeddings.shape[1]  # 1536
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index (Inner Product on normalized = Cosine)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.urls = urls
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for similar vectors."""
        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, top_k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'url': self.urls[idx],
                'score': float(score)  # Cosine similarity
            })
        return results
```

### Scaling Considerations

| Vectors | Recommended Index | Memory | Search Time |
|---------|-------------------|--------|-------------|
| < 10K | IndexFlatIP | ~60 MB | < 1ms |
| 10K - 100K | IndexFlatIP | ~600 MB | < 10ms |
| 100K - 1M | IndexIVFFlat | ~600 MB | < 10ms |
| 1M - 10M | IndexIVFPQ | ~1 GB | < 50ms |
| 10M - 100M | IndexHNSW + PQ | ~10 GB | < 100ms |
| 100M+ | Distributed (Milvus, Pinecone) | Varies | Varies |

---

# 5. LLMs & Prompt Engineering

## BFS (Simple Explanation)
"Large Language Models predict the next word based on patterns learned from massive text datasets. Prompt engineering is the art of crafting inputs that get the best outputs from these models."

## DFS (Technical Deep-Dive)

### How LLMs Work (Transformer Architecture)

```
Input: "The capital of France is"
              ↓
┌─────────────────────────────────────────┐
│            TOKENIZATION                  │
│  "The" "capital" "of" "France" "is"     │
│    ↓       ↓      ↓      ↓      ↓       │
│   [464]  [3139]  [286]  [4881]  [318]   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         TOKEN EMBEDDINGS                 │
│  Each token ID → learned vector          │
│  [464] → [0.1, -0.2, ..., 0.3]          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│      POSITIONAL ENCODING                 │
│  Add position information                │
│  Token 1, Token 2, Token 3, ...         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│    TRANSFORMER BLOCKS (×96 for GPT-4)   │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │    MULTI-HEAD SELF-ATTENTION       │ │
│  │    Each token attends to others    │ │
│  │    "France" ← pays attention to →  │ │
│  │    "capital", "of"                 │ │
│  └────────────────────────────────────┘ │
│              ↓                           │
│  ┌────────────────────────────────────┐ │
│  │    FEED-FORWARD NETWORK            │ │
│  │    2 linear layers + activation    │ │
│  └────────────────────────────────────┘ │
│              ↓                           │
│  ┌────────────────────────────────────┐ │
│  │    LAYER NORMALIZATION             │ │
│  └────────────────────────────────────┘ │
│                                          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         OUTPUT PROJECTION                │
│  Final hidden state → vocabulary logits  │
│  [0.001, 0.002, ..., 0.95, ...]         │
│                            ↑             │
│                         "Paris"          │
└─────────────────────────────────────────┘
              ↓
Output: "Paris"
```

### Self-Attention Mechanism (The Key Innovation)

```
Query, Key, Value (Q, K, V):

For each token, we create 3 vectors:
- Query (Q): "What am I looking for?"
- Key (K): "What do I contain?"
- Value (V): "What do I offer?"

Attention Formula:
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Example for "capital" in "The capital of France is":
- Q_capital asks: "What noun am I describing?"
- K_France answers: "I'm a country name"
- High attention score: capital → France
- V_France contributes to capital's representation
```

```
Attention Matrix Visualization:

             The  capital  of  France  is
        ┌─────────────────────────────────┐
The     │ 0.8   0.1     0.05  0.03   0.02│
capital │ 0.1   0.2     0.1   0.5    0.1 │  ← "capital" attends to "France"
of      │ 0.1   0.3     0.2   0.3    0.1 │
France  │ 0.05  0.4     0.1   0.4    0.05│
is      │ 0.1   0.2     0.05  0.3    0.35│
        └─────────────────────────────────┘
```

### Multi-Head Attention
```
Instead of one attention, run multiple in parallel:

Head 1: Focuses on syntactic relationships
Head 2: Focuses on semantic similarity  
Head 3: Focuses on positional patterns
...
Head 12: Focuses on something else learned

Then concatenate and project:
MultiHead = Concat(head_1, ..., head_h) × W_O
```

### Models I Use

| Model | Parameters | Context | Cost | Use Case |
|-------|------------|---------|------|----------|
| GPT-4o | ~1.8T (rumored) | 128K | $5/1M in | Complex reasoning |
| GPT-4o-mini | Smaller | 128K | $0.15/1M in | My lesson generation |
| text-embedding-3-small | ~100M | 8K | $0.02/1M | My embeddings |

### Prompt Engineering Techniques

#### 1. System Prompts (Role Setting)
```python
# From lesson_service.py
system_prompt = """You are an expert educator who transforms complex 
research papers into beginner-friendly lessons. 

Your explanations should:
- Use simple analogies
- Build concepts progressively
- Include concrete examples
- Avoid jargon unless explained
"""
```

#### 2. Few-Shot Learning
```python
prompt = """
Example 1:
Paper section: "We utilize transformer-based architecture..."
Lesson: "Think of transformers like a smart reader that can look at 
all words at once, rather than reading left to right..."

Example 2:
Paper section: "The attention mechanism computes..."
Lesson: "Attention is like a spotlight - it helps the model focus 
on the most relevant words..."

Now convert this section:
Paper section: {actual_section}
Lesson:
"""
```

#### 3. Chain of Thought (CoT)
```python
prompt = """
Let's think step by step:
1. First, identify the main concept in this section
2. Then, find a simple analogy
3. Next, explain the technical details using the analogy
4. Finally, provide a concrete example

Section: {paper_section}
"""
```

#### 4. Structured Output
```python
prompt = """
Convert this paper section into a lesson.

Output format:
{
  "main_concept": "...",
  "simple_explanation": "...",
  "analogy": "...",
  "example": "...",
  "key_takeaway": "..."
}

Section: {paper_section}
"""
```

### My Actual Prompt from the Project
```python
# From lesson_service.py
def _build_prompt(self, section: PaperSection, difficulty: str) -> str:
    difficulty_instructions = {
        'beginner': 'Use simple language, analogies, and avoid jargon.',
        'intermediate': 'Assume basic ML knowledge, explain advanced concepts.',
        'advanced': 'Be technical, include mathematical details.'
    }
    
    return f"""
    You are an expert AI educator. Convert this research paper section 
    into an educational lesson.
    
    Difficulty: {difficulty}
    Instructions: {difficulty_instructions[difficulty]}
    
    Section Name: {section.name}
    Section Content: {section.content}
    
    Create an engaging, clear explanation that:
    1. Introduces the concept
    2. Explains WHY it matters
    3. Provides examples or analogies
    4. Summarizes key points
    
    Write in markdown format.
    """
```

### Temperature and Other Parameters

```python
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    temperature=0.7,      # 0=deterministic, 1=creative, 2=chaotic
    max_tokens=1000,      # Max output length
    top_p=0.9,            # Nucleus sampling (alternative to temperature)
    frequency_penalty=0.5, # Reduce repetition
    presence_penalty=0.5,  # Encourage new topics
)
```

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| temperature | Focused, deterministic | Creative, varied |
| top_p | Conservative word choices | More diverse vocabulary |
| frequency_penalty | May repeat phrases | Avoids repetition |
| presence_penalty | Stays on topic | Explores new topics |

---

# 6. System Architecture

## BFS (Simple Explanation)
"The system has a React frontend that talks to a FastAPI backend. The backend orchestrates several services: embedding service for vector operations, paper service for PDF handling, and lesson service for content generation."

## DFS (Technical Deep-Dive)

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    React + TypeScript                            │   │
│  │  • Hero component (input)                                        │   │
│  │  • LessonDisplay (output)                                        │   │
│  │  • Theme switching (dark/light)                                  │   │
│  │  • Framer Motion animations                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP/REST (JSON)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            API LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                           │   │
│  │                                                                   │   │
│  │  Routes:                                                          │   │
│  │  • POST /api/v1/teach         → Generate lesson                  │   │
│  │  • POST /api/v1/teach/stream  → Stream lesson (SSE)              │   │
│  │  • POST /api/v1/leetcode/random → Get coding problem             │   │
│  │  • GET  /health               → Health check                      │   │
│  │                                                                   │   │
│  │  Middleware:                                                      │   │
│  │  • CORS                       → Cross-origin requests            │   │
│  │  • Request timing             → Performance monitoring           │   │
│  │  • Exception handlers         → Structured error responses       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          SERVICE LAYER                                   │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ TeachingService  │  │  PaperService    │  │  LessonService   │      │
│  │                  │  │                  │  │                  │      │
│  │ • Orchestrates   │  │ • FAISS search   │  │ • GPT generation │      │
│  │   full pipeline  │  │ • PDF download   │  │ • Prompt building│      │
│  │ • Coordinates    │  │ • GROBID parsing │  │ • Streaming      │      │
│  │   all services   │  │ • Section extract│  │                  │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│           │                     │                     │                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │EmbeddingService  │  │ LeetCodeService  │  │  CacheService    │      │
│  │                  │  │                  │  │                  │      │
│  │ • OpenAI embed   │  │ • Fetch problems │  │ • LRU memory     │      │
│  │ • FAISS index    │  │ • Parse HTML     │  │ • File persist   │      │
│  │ • Vector search  │  │ • Filter by diff │  │ • TTL expiry     │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SERVICES                                 │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   OpenAI     │  │    arXiv     │  │   GROBID     │  │  LeetCode  │  │
│  │              │  │              │  │              │  │            │  │
│  │ • Embeddings │  │ • Paper PDFs │  │ • PDF parse  │  │ • Problems │  │
│  │ • Chat API   │  │ • Metadata   │  │ • Section    │  │ • GraphQL  │  │
│  │              │  │              │  │   extraction │  │            │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Request Flow (Detailed)

```
User types: "Explain attention mechanisms"
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. FRONTEND: Hero.tsx                                           │
│    • User submits query                                         │
│    • App.tsx calls generateLesson(request)                      │
│    • Shows loading overlay                                      │
└─────────────────────────────────────────────────────────────────┘
                        │ POST /api/v1/teach
                        │ {"query": "Explain attention...", ...}
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. API ROUTE: routes/teach.py                                   │
│    • Validate request with Pydantic                             │
│    • Call teaching_service.teach()                              │
│    • Return LessonResponse                                      │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. TEACHING SERVICE: services/teaching_service.py               │
│    • Orchestrate the full pipeline                              │
│    • Log request start                                          │
└─────────────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌───────────────────┐         ┌─────────────────────────┐
│ 4a. EMBED QUERY   │         │ 4b. CHECK CACHE         │
│ embedding_service │         │ cache_service           │
│ .embed(query)     │         │ .get("lessons", key)    │
│                   │         │                         │
│ → OpenAI API      │         │ Cache miss → continue   │
│ → 1536-dim vector │         │ Cache hit → return early│
└───────────────────┘         └─────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. FAISS SEARCH: embedding_service.search()                     │
│    • Load index (231 vectors)                                   │
│    • Normalize query vector                                     │
│    • index.search(query, k=1)                                   │
│    • Return: paper URL + similarity score                       │
│                                                                  │
│    Result: arxiv.org/abs/1706.03762 (Attention paper), score=0.72│
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. FETCH PAPER: paper_service.get_paper(url)                    │
│                                                                  │
│    a) Fetch metadata from arXiv API                             │
│       → Title, authors, abstract, date                          │
│                                                                  │
│    b) Download PDF                                               │
│       → GET arxiv.org/pdf/1706.03762.pdf                        │
│                                                                  │
│    c) Parse with GROBID                                          │
│       → POST to GROBID cloud service                            │
│       → Returns TEI-XML                                          │
│       → Extract sections: abstract, introduction, methods...    │
│                                                                  │
│    Result: ParsedPaper with 24 sections                         │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. GENERATE LESSONS: lesson_service.generate_lesson()           │
│                                                                  │
│    For each section (limited to max_sections=5):                │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ a) Build prompt with section content                     │  │
│    │ b) Call OpenAI GPT-4o-mini                               │  │
│    │ c) Parse response into LessonFragment                    │  │
│    │ d) Calculate read time                                   │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│    Result: Lesson with 5 fragments, 15 min total read time     │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. CACHE RESULT: cache_service.set("lessons", key, lesson)      │
│    • Store in LRU memory cache                                  │
│    • Persist to file system                                     │
│    • TTL: 24 hours                                              │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. RETURN RESPONSE                                              │
│    {                                                             │
│      "success": true,                                           │
│      "lesson": {                                                │
│        "paper_id": "1706.03762",                                │
│        "paper_title": "Attention Is All You Need",             │
│        "fragments": [...],                                      │
│        "total_read_time": 15                                    │
│      },                                                         │
│      "processing_time_ms": 45000                                │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. FRONTEND: Display lesson                                    │
│     • LessonDisplay.tsx renders                                 │
│     • Table of contents                                         │
│     • Collapsible sections                                      │
│     • Markdown rendering                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Singleton** | All services (`get_*_service()`) | One instance, shared state |
| **Facade** | TeachingService | Simple interface to complex subsystem |
| **Strategy** | Difficulty levels | Different prompts based on level |
| **Factory** | Pydantic models | Create validated objects |
| **Repository** | CacheService | Abstract data access |
| **Dependency Injection** | Services init | Loose coupling, testability |

---

# 7. Backend Deep Dive

## FastAPI Fundamentals

### Why FastAPI?
```python
# Automatic validation
@app.post("/teach")
async def teach(request: LessonRequest) -> LessonResponse:
    # request is already validated by Pydantic
    # Response is serialized automatically
    pass

# Compare to Flask:
@app.route("/teach", methods=["POST"])
def teach():
    data = request.get_json()  # No validation
    # Manual validation needed
    # Manual serialization needed
```

### Async/Await
```python
# Synchronous (blocking)
def fetch_paper(url):
    response = requests.get(url)  # Blocks entire server
    return response.text

# Asynchronous (non-blocking)
async def fetch_paper(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()  # Other requests can run

# Why it matters:
# 100 concurrent requests, each takes 1 second:
# Sync: 100 seconds total
# Async: ~1 second total (all run in parallel)
```

### Pydantic Models
```python
from pydantic import BaseModel, Field, field_validator

class LessonRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    difficulty: Literal['beginner', 'intermediate', 'advanced'] = 'beginner'
    max_sections: int = Field(default=5, ge=1, le=20)
    
    @field_validator('query')
    @classmethod
    def clean_query(cls, v):
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain attention mechanisms",
                "difficulty": "beginner"
            }
        }
```

### Error Handling
```python
# Custom exceptions
class PaperNotFoundError(Exception):
    status_code = 404
    detail = "Paper not found"

class GROBIDError(Exception):
    status_code = 502
    detail = "GROBID service unavailable"

# Global exception handler
@app.exception_handler(PaperNotFoundError)
async def paper_not_found_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

### Middleware
```python
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Processing-Time"] = f"{duration:.3f}s"
    return response
```

---

# 8. Frontend Deep Dive

## React + TypeScript

### Component Architecture
```
App.tsx                          # Root component, state management
├── ThemeProvider                # Context for dark/light mode
├── Header.tsx                   # Navigation, theme toggle
├── Hero.tsx                     # Input form, suggestions
├── LessonDisplay.tsx            # Modal with lesson content
│   └── LessonSection.tsx        # Collapsible section
└── ProblemDisplay.tsx           # LeetCode problem modal
```

### State Management
```tsx
// Using React's built-in state (no Redux needed for this scale)
type ViewState = 
  | { type: 'home' }
  | { type: 'loading'; message: string }
  | { type: 'lesson'; lesson: Lesson }
  | { type: 'error'; message: string }

function App() {
  const [viewState, setViewState] = useState<ViewState>({ type: 'home' })
  
  // State machine pattern
  const handleSubmit = async (query: string) => {
    setViewState({ type: 'loading', message: 'Searching...' })
    try {
      const lesson = await generateLesson({ query })
      setViewState({ type: 'lesson', lesson })
    } catch (error) {
      setViewState({ type: 'error', message: error.message })
    }
  }
}
```

### Custom Hooks
```tsx
// useTheme.tsx
function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('system')
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('light')
  
  useEffect(() => {
    // Listen to system preference
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      if (theme === 'system') {
        setResolvedTheme(mediaQuery.matches ? 'dark' : 'light')
      }
    }
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [theme])
  
  return { theme, setTheme, resolvedTheme }
}
```

### Tailwind CSS
```tsx
// Utility-first approach
<button className={cn(
  // Base styles
  "px-4 py-2 rounded-xl font-medium transition-all",
  // Conditional styles
  isActive 
    ? "bg-primary-500 text-white shadow-lg" 
    : "bg-gray-100 text-gray-600 hover:bg-gray-200",
  // Passed-in styles
  className
)}>
  {children}
</button>

// cn() utility merges Tailwind classes intelligently
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

function cn(...inputs) {
  return twMerge(clsx(inputs))
}
```

### Framer Motion Animations
```tsx
<motion.div
  initial={{ opacity: 0, y: 20 }}      // Start state
  animate={{ opacity: 1, y: 0 }}       // End state
  exit={{ opacity: 0, y: -20 }}        // Exit state
  transition={{ duration: 0.3 }}       // Timing
>
  {content}
</motion.div>

// AnimatePresence for exit animations
<AnimatePresence>
  {showModal && <Modal />}
</AnimatePresence>
```

---

# 9. DevOps & Deployment

## Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/streaming-support
# Make changes
git add .
git commit -m "Add SSE streaming for lessons"
git push origin feature/streaming-support
# Create PR, review, merge
```

## Railway Deployment
```
GitHub Push → Railway Webhook → Build → Deploy

Build Process:
1. Clone repo
2. Detect language (Python/Node)
3. Install dependencies
4. Run build command
5. Start application

Environment Variables:
- OPENAI_API_KEY (secret)
- GROBID_URL
- USE_GROBID=true
```

## Docker (If You Want to Discuss)
```dockerfile
# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# 10. Common Interview Questions

## About Your Project

### Q: "Walk me through your project."
**Answer Framework:**
1. Problem: "Research papers are hard to understand"
2. Solution: "RAG-based system that finds and teaches from real papers"
3. Tech: "FastAPI backend, React frontend, FAISS for search"
4. Impact: "Users can learn complex topics with cited sources"

### Q: "Why did you choose RAG over fine-tuning?"
**Answer:**
- Cost: No training compute needed
- Flexibility: Add papers without retraining
- Transparency: Can cite sources
- Freshness: Always uses latest papers

### Q: "How does your similarity search work?"
**Answer:**
1. Convert text to 1536-dim vector using OpenAI embeddings
2. Normalize vectors for cosine similarity
3. FAISS IndexFlatIP for exact nearest neighbor search
4. Return paper with highest similarity score

### Q: "What would you do differently with more time?"
**Answer Ideas:**
- Add hybrid search (semantic + keyword)
- Implement query rewriting
- Add user accounts and history
- Support more document types
- Add evaluation metrics

### Q: "How would you scale this?"
**Answer:**
- Replace FAISS with Pinecone/Weaviate for managed vector DB
- Add Redis for caching
- Use Kubernetes for container orchestration
- Implement rate limiting with Redis
- Add CDN for static assets

## Technical Concepts

### Q: "Explain how transformers work."
**Answer:**
"Transformers process all tokens in parallel using self-attention. Each token creates Query, Key, Value vectors. Attention scores are computed as softmax(QK^T/√d). This lets the model learn which words are relevant to each other, regardless of distance. Multi-head attention runs this multiple times to capture different relationships."

### Q: "What's the difference between cosine similarity and Euclidean distance?"
**Answer:**
"Cosine measures the angle between vectors (direction), while Euclidean measures absolute distance (magnitude). Cosine is better for text because we care about semantic direction, not magnitude. Two documents about the same topic should be similar even if one is longer."

### Q: "How does GROBID extract sections from PDFs?"
**Answer:**
"GROBID uses CRF (Conditional Random Fields) models trained on academic papers. It identifies structural elements like title, abstract, headers, paragraphs, and figures based on layout and text features. The output is TEI-XML which I parse to extract clean sections."

### Q: "What is prompt engineering?"
**Answer:**
"Prompt engineering is crafting inputs to get desired outputs from LLMs. Key techniques include: role setting (system prompts), few-shot examples, chain-of-thought reasoning, and structured output formats. I use difficulty-specific prompts that adjust language complexity based on user level."

## Behavioral

### Q: "Tell me about a challenge you faced."
**Example Answer:**
"Deploying to Railway, the FAISS index wasn't being found. I discovered the paths were relative, but Railway runs from a different directory. I fixed it by making paths absolute based on the project root using `Path(__file__).parent.parent`. This taught me to always consider the deployment environment during development."

### Q: "What did you learn building this?"
**Answer Ideas:**
- RAG architecture and its trade-offs
- Vector similarity search at scale
- Full-stack deployment with environment management
- Prompt engineering for educational content
- The importance of error handling and logging

---

# 🎓 Study Checklist

Before your interview, make sure you can:

## Concepts
- [ ] Explain RAG in simple terms and technically
- [ ] Draw the system architecture from memory
- [ ] Explain embeddings and similarity search
- [ ] Describe how FAISS indexes work
- [ ] Explain transformer attention mechanism
- [ ] Discuss prompt engineering techniques

## Your Code
- [ ] Walk through the request flow
- [ ] Explain each service's responsibility
- [ ] Discuss design patterns used
- [ ] Explain your error handling strategy
- [ ] Describe your caching approach

## Improvements
- [ ] List 3 ways to improve accuracy
- [ ] List 3 ways to improve performance
- [ ] List 3 ways to scale the system
- [ ] Discuss monitoring/observability additions

---

**Good luck with your interviews! 🚀**

Remember: It's not just about knowing the answers—it's about showing your thinking process and genuine curiosity for the technology.
