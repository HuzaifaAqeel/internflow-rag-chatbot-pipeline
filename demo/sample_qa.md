# RAG Chatbot — Sample Q&A

## Q1: What are the three main paradigms of machine learning?

**A:** There are three main paradigms. What is Machine Learning? Supervised Learning Supervised learning is the most widely used machine learning paradigm.

Sources:
- What is Machine Learning? (`01_what_is_ml.md`#chunk0, score=0.1892)
- What is Machine Learning? (`01_what_is_ml.md`#chunk0, score=0.1892)
- Supervised Learning (`02_supervised_learning.md`#chunk0, score=0.0533)

_latency=5.3 ms, backend=extractive, embedder=tfidf-fallback_

## Q2: What is overfitting and how can it be prevented?

**A:** Overfitting and Underfitting Overfitting and underfitting describe the two failure modes of model training. Techniques to fight overfitting include cross-validation, regularization, and collecting more training data. A key danger is overfitting, where the model memorizes the training data but fails on new data.

Sources:
- Overfitting and Underfitting (`05_overfitting_underfitting.md`#chunk0, score=0.1014)
- Supervised Learning (`02_supervised_learning.md`#chunk0, score=0.0686)
- Supervised Learning (`02_supervised_learning.md`#chunk0, score=0.0686)

_latency=1.9 ms, backend=extractive, embedder=tfidf-fallback_

## Q3: How does a random forest reduce overfitting?

**A:** Ensembling is more powerful: a random forest trains hundreds of trees on bootstrap samples of the data, each tree seeing only a random subset of features at each split, and averages their votes. Overfitting and Underfitting Overfitting and underfitting describe the two failure modes of model training. Techniques to fight overfitting include cross-validation, regularization, and collecting more training data.

Sources:
- Decision Trees and Random Forests (`08_decision_trees.md`#chunk0, score=0.0832)
- Overfitting and Underfitting (`05_overfitting_underfitting.md`#chunk0, score=0.0866)
- Supervised Learning (`02_supervised_learning.md`#chunk0, score=0.0528)

_latency=1.6 ms, backend=extractive, embedder=tfidf-fallback_

## Q4: What does a vector database like FAISS do?

**A:** Indexing stores the vectors in a vector database such as FAISS or ChromaDB. Sentence embeddings extend this idea to whole sentences or paragraphs: models like all-MiniLM-L6-v2 or E5 map any text into a fixed-size vector (for example, 384 dimensions) where semantic similarity corresponds to geometric closeness, usually measured with cosine similarity. Embedding converts each chunk into a vector with a sentence-embedding model.

Sources:
- Retrieval-Augmented Generation (RAG) (`11_rag.md`#chunk0, score=0.1082)
- Embeddings and Vector Search (`10_embeddings.md`#chunk0, score=0.057)
- Retrieval-Augmented Generation (RAG) (`11_rag.md`#chunk0, score=0.1082)

_latency=1.5 ms, backend=extractive, embedder=tfidf-fallback_

## Q5: What are the five stages of a RAG pipeline?

**A:** A typical RAG pipeline has five stages. Evaluation of RAG systems measures both stages: retrieval hit-rate (is the answer contained in the top-k chunks?), and answer faithfulness (does the generated answer stay consistent with the retrieved context?). Retrieval-Augmented Generation (RAG) Retrieval-Augmented Generation (RAG) combines a retriever with a generator to answer questions from external knowledge.

Sources:
- Retrieval-Augmented Generation (RAG) (`11_rag.md`#chunk0, score=0.1544)
- Retrieval-Augmented Generation (RAG) (`11_rag.md`#chunk1, score=0.0883)
- Retrieval-Augmented Generation (RAG) (`11_rag.md`#chunk0, score=0.1544)

_latency=1.4 ms, backend=extractive, embedder=tfidf-fallback_

## Q6: When is accuracy a misleading metric?

**A:** Accuracy — the fraction of correct predictions — is intuitive but misleading on imbalanced data: a model that always predicts the majority class can score 95% accuracy while being useless. Evaluation Metrics for Classification Choosing the right metric is as important as choosing the right model, because the metric defines what "good" means. For classification we report accuracy, precision, recall, and F1-score; for regression we use mean squared error (MSE) or mean absolute error (MAE).

Sources:
- Evaluation Metrics for Classification (`12_evaluation_metrics.md`#chunk0, score=0.159)
- Evaluation Metrics for Classification (`12_evaluation_metrics.md`#chunk0, score=0.159)
- Supervised Learning (`02_supervised_learning.md`#chunk0, score=0.0285)

_latency=1.5 ms, backend=extractive, embedder=tfidf-fallback_
