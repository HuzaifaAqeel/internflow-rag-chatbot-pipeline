# Sample Corpus — how it was created

The `corpus/` folder contains **14 short markdown documents** (2,614 words
total) on **machine learning fundamentals**, written specifically as sample
data for this project. They are **not scraped from anywhere** — each document
was hand-written to cover one topic cleanly so that retrieval quality can be
evaluated with known ground truth:

| # | File | Topic |
|---|------|-------|
| 1 | `01_what_is_ml.md` | What is Machine Learning? |
| 2 | `02_supervised_learning.md` | Supervised Learning |
| 3 | `03_unsupervised_learning.md` | Unsupervised Learning |
| 4 | `04_neural_networks.md` | Neural Networks |
| 5 | `05_overfitting_underfitting.md` | Overfitting and Underfitting |
| 6 | `06_cross_validation.md` | Cross-Validation |
| 7 | `07_gradient_descent.md` | Gradient Descent |
| 8 | `08_decision_trees.md` | Decision Trees and Random Forests |
| 9 | `09_nlp_basics.md` | Natural Language Processing Basics |
| 10 | `10_embeddings.md` | Embeddings and Vector Search |
| 11 | `11_rag.md` | Retrieval-Augmented Generation (RAG) |
| 12 | `12_evaluation_metrics.md` | Evaluation Metrics for Classification |
| 13 | `13_feature_engineering.md` | Feature Engineering |
| 14 | `14_regularization.md` | Regularization |

## Limitations (honest)

- The corpus is **synthetic and small** — written by the project author, not a
  real production knowledge base. Retrieval scores on it do not transfer to
  noisy real-world documents.
- Documents are clean markdown with no tables, images, PDFs, or scanned text —
  the ingestion path for those formats is **not** implemented.
- Topics were chosen to be mutually distinguishable, which flatters retrieval
  metrics versus a corpus with heavy topic overlap.
