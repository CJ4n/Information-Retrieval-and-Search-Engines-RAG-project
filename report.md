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
#### term vocabulary:
 1. I apply vocabulary preprocessing: lammentizer, to lower case, remove punctuation, remove english stop words and tokenize. Also I added some custom words that are too generic to bring in any usefull info, but clutter a lot the 2-grams embeddings (eg. add, added).
 Next i use two TF-IDF vectorizesr, 1-gram and 2-gram, so i can capture single words expression and two-words expressionl. The reason for not using just one vectorizesr using both 1-gram and 2-gram, is taht 2-grams dominated the vocabulary, but most of the times they were very generic, and were just meaningless phraes that did not bring any usefull info
 2. Not exactly, I do not set fixed number of features, but I do chose top 30% of features. The reason for that is, when I use N-grams (in my case 1-gram and 2-gram) I get a lot of useless features (eg. add X, add Y, add Z). By taking only a subset of all possible features, I try to get rid of useless features. 

 On top of that, I also do not chose features that appear in 'too many' documents, ie. >70% of all documents, so I do not include terms that are meaningless as they almost always appear and do not chose feature that appear only in few documents, ie. <10 documents, so i do not include terms that may be just artifact, eg. types, URLs, etc.
3. I use 2-gram to capture mulit-word (2-word) expressions and apply very aggressive threesholding in TFIDF to select only raelly relevant ones (TODO add exact params)
#### document embeddeing
 1. TODO
 2. TODO
 3. TODO

#### retrieval
1. I tried out cosine similatiy and Euclidean distance. In my experiment it turned out that cosine similatiy works better. Also my research indicated that cosine similatiy is the prefered option for TF-IDF embeddeings, as it ignores magunitude, and measure only angular difference, which is prefered for tasks of retrieving recipies.
2. TODO
3. TODO
4. TODO

#### qualitative analysis (IR)
1. TODO

#### Propt
1. TODO
2. TODO

#### qualiative analysis (LM)
1. TODO
2. TODO
3. TODO
4. TODO
