Lab 04: RAG System with Evaluation
Objective
Build a complete RAG system for querying a codebase, including proper evaluation.

Time Allotted: 1 hour 45 minutes

Learning Goals
Implement a RAG pipeline from scratch
Use appropriate chunking for code
Build an evaluation framework
Understand retrieval metrics
What You'll Build
A codebase Q&A system that:

Indexes code files with embeddings
Retrieves relevant code for questions
Generates answers grounded in code
Evaluates retrieval and generation quality
┌─────────────────────────────────────────────────────────────┐
│ Codebase RAG System │
├─────────────────────────────────────────────────────────────┤
│ │
│ INDEXING │
│ ──────── │
│ Code Files → Chunk → Embed → Store (Vector DB) │
│ │
│ QUERYING │
│ ──────── │
│ Question → Embed → Search → Retrieve → Generate Answer │
│ │
│ EVALUATION │
│ ────────── │
│ Test Questions → RAG → Compare to Ground Truth → Metrics │
│ │
└─────────────────────────────────────────────────────────────┘
Requirements
Core Functionality
Code Chunker: splits code files into meaningful chunks (by function/class for Python, generic for others)
Vector Store: stores and retrieves code chunks using embeddings (ChromaDB for Python, in-memory for TypeScript)
RAG Pipeline:
POST /index/files endpoint to index code files
POST /query endpoint to ask questions about indexed code
POST /evaluate endpoint to run evaluation suite
Evaluation Framework implementing:
Precision@K
Recall@K
MRR (Mean Reciprocal Rank)
LLM-as-judge for generation quality
Frontend Requirements
Web interface with two sections: Indexing and Querying
File upload area for indexing code files
Chat-like interface for asking questions about indexed code
Results showing answer + source code snippets used
Evaluation metrics display panel
Responsive design
Language Choice
Aspect Python TypeScript
Vector DB ChromaDB (persistent) In-memory (demo)
Embeddings sentence-transformers (free) or OpenAI OpenAI API
Framework FastAPI Hono
Deliverables
Working RAG system with code indexing
Smart code-aware chunking
Evaluation framework with retrieval metrics
LLM-as-judge generation evaluation
Deployed to Railway/Vercel
Evaluation dataset (10+ examples)
Web frontend for code indexing and Q&A
Application deployed to Vercel/Railway/Render (provide URL)
Extension Challenges
Hybrid Search: Add BM25 keyword search alongside vector search
Reranking: Add a reranking step to improve retrieval
Caching: Cache embeddings and query results
Multiple Codebases: Support querying across multiple indexed repos
