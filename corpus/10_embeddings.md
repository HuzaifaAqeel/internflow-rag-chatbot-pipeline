# Embeddings and Vector Search

An embedding is a dense numerical vector that represents an object — a word, sentence, image, or user — such that similar objects have nearby vectors. Sentence embeddings extend this idea to whole sentences or paragraphs: models like all-MiniLM-L6-v2 or E5 map any text into a fixed-size vector (for example, 384 dimensions) where semantic similarity corresponds to geometric closeness, usually measured with cosine similarity.

Vector databases store these embeddings together with their source documents and answer similarity queries at scale. FAISS (Facebook AI Similarity Search) is a library for efficient nearest-neighbour search over dense vectors, supporting exact search with IndexFlatIP and approximate search with IVF or HNSW indexes for millions of vectors. Alternatives include ChromaDB, Weaviate, Pinecone, and Qdrant, which add metadata filtering, persistence, and distributed serving.

Embeddings power semantic search, recommendation, deduplication, and clustering. In a RAG system, the user's question is embedded with the same model as the documents, the vector store returns the top-k most similar chunks, and those chunks become the context for answer generation. Keeping the query and document encoders identical is essential — mismatched embeddings silently destroy retrieval quality.
