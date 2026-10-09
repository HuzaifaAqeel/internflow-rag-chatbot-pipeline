# Natural Language Processing Basics

Natural Language Processing (NLP) is the branch of machine learning concerned with text. Raw text must first be tokenized — split into words, subwords, or characters — because models operate on numbers, not strings. Classical pipelines then clean the text: lowercasing, removing punctuation and stop words, and reducing words to stems or lemmas.

To feed text into a model it must be vectorized. Bag-of-words counts word occurrences but ignores order. TF-IDF (term frequency–inverse document frequency) down-weights words that appear in many documents, so distinctive words matter more. Dense word embeddings such as Word2Vec, GloVe, and FastText map each word to a vector where similar meanings sit close together; they are learned from large corpora by predicting surrounding words.

Modern NLP is dominated by the Transformer architecture, which uses self-attention to weigh the relevance of every token to every other token. Pretrained transformers like BERT (for understanding tasks) and GPT (for generation) are fine-tuned on small labelled datasets and achieve state-of-the-art results on classification, question answering, summarization, and translation.
