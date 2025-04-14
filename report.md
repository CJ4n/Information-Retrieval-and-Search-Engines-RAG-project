1. draw architecture
2. investiagate how can i chose which terms should i Use
3. run evalute on different embeddeing fields

### par 1

#### architecture

 1. TODO
 2. Query - user input text
 Preprocessor - process query (remove stopwords, punctuation, transfoer to lower case, tokenize, remove numbers) (the same preprocessing as for recipy dataset)
 TF-IDF uni-vectorizer - apply TFIDF with 1-gram to extract features
 TF-idf bi-vectorizer - apply TFIDF with 2-gram to extract compound terms (eg. olive oil)
 kNN+ threeshold - select K best matching documents and apply threshold so you don't retrive very irrelevant document (combine uni and bi features)
 retrieved documents - set of retriebed documents
Prompt - system prompot for LLM, combined with user's query and retrieved documenst (in original form (not preprocessed))
LLM-tokenizer - tokenizer to tokenize prompt to work with LLM
LLM-generation - LLM model taht generates answer
LLM-decode - use tokenizer to decode LLM generated answer
answer - final answer provided to user

#### term vocabulary

 1. I apply vocabulary preprocessing: lammentizer, to lower case, remove punctuation, remove english stop words and tokenize. Also I added some custom words that are too generic to bring in any usefull info, but clutter a lot the 2-grams embeddings (eg. add, added).
 Next i use two TF-IDF vectorizesr, 1-gram and 2-gram, so i can capture single words expression and two-words expressionl. The reason for not using just one vectorizesr using both 1-gram and 2-gram, is taht 2-grams dominated the vocabulary, but most of the times they were very generic, and were just meaningless phraes that did not bring any usefull info
 2. Yes and no. Since I have two vecotrizeres, for 1-gram and 2-gram, i do not set fixed size of vocabulary for 1-gram vectorizesr, but i use min_df and max_df to constrain size of vocabulary. However, I set max features for 2-gram, as it generted more then 200,000 features, so i take only 10,000 of them, but also set min_df and max_df, to reduce noise from very infrequent 2-gram features, and only inclode those that are really common and relvant, but also not so common, that they appear in all recipies.
 3. I use 2-gram to capture mulit-word (2-word) expressions and apply very aggressive threesholding in TFIDF to select only raelly relevant ones (TODO add exact params)

#### document embeddeing

 1. I experimented with different combination of fields (including just single field, but also combinations of two different fields) and found that using all fields (name, description, ingredients and steps) gives the best results. Which seems to make sense, as more information is available for the retrieval model to work with, thus it will produce more accurate results.
 2. I use precisly the same pipeline as for embedding documents. First I apply preprocessing the the query and then two combined TF-IDF vectorizesr to embed query.
 3. Since no words from vocabulary are in the query, I get zero vector for both 1-gram and 2-gram TF-IDF vectorizer. Cosine similarity score is 0.0, so when i apply threshold, I get empty set of documents.

#### retrieval

I used grid search to find best K and threshold for kNN.

1. I tried out cosine similatiy and Euclidean distance. In my experiment it turned out that cosine similatiy works mutch better. Also my research indicated that cosine similatiy is the prefered option for TF-IDF embeddeings, as it ignores magnitude, and measure only angular difference, which is prefered for tasks of retrieving recipies.
2. I do and do not vary number of returned documents. On top of fixed K returned documents I also apply threshold, so I do not return documents that are too far away from the query. I use grid search to find best K and threshold.
3. {'macro_precision': np.float64(0.13003157750621686),
 'macro_recall': np.float64(0.200627617143981),
 'macro_f1': np.float64(0.1257867808600322),
 'micro_precision': 0.1279229711141678,
 'micro_recall': 0.19057377049180327,
 'micro_f1': 0.1530864197530864,
 'map': 0.08646612040391587}
4. AP = \frac{1}{RD} \sum_{k=1}^{K} P(k) \cdot r(k)
Where 𝑅𝐷 is the number of relevant documents for the query, K is the number of retrieved documents, 𝑃(𝑘) is the precision at 𝑘, and 𝑟(𝑘) is the relevance of the 𝑘 𝑡 ℎ retrieved document (0 if not relevant, and 1 if relevant). Mean Average Precision is this AP metric, averaged across the set of queries
{'map': 0.08646612040391587}

#### qualitative analysis (IR)

1. TODO

#### Prompt

1. TODO
2. TODO

#### qualiative analysis (LM)

1. TODO
2. TODO
3. TODO
4. TODO
