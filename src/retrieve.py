import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

from evaluation import *
from metrics import *

DEBUG = False


def retrieve_documents(
    query_text, recipies, recipe_ids, k, threshold, vec_uni, vec_bi, X_uni, X_bi
):
    if len(recipies) != len(recipe_ids):
        raise ValueError("Recipes and recipe_ids must have the same length")
    if k is None and threshold is None:
        raise ValueError("Either k or threshold must be specified")

    query = preprocess_text(query_text)
    if DEBUG:
        print("PREPROCEDDE QUERY: ", query)
    q_uni = vec_uni.transform([query])
    q_bi = vec_bi.transform([query])
    q_all = hstack([q_uni, q_bi])

    X_all = hstack([X_uni, X_bi])
    if DEBUG:
        uni_feature_names = vec_uni.get_feature_names_out()
        bi_feature_names = vec_bi.get_feature_names_out()

        q_uni_indices = q_uni.nonzero()[1]
        q_bi_indices = q_bi.nonzero()[1]

        print("Query terms (unigrams):", [uni_feature_names[i] for i in q_uni_indices])
        print("Query terms (bigrams):", [bi_feature_names[i] for i in q_bi_indices])

    cosine_similarities = cosine_similarity(q_all, X_all).flatten()

    results = [
        (recipies[i], recipe_ids[i], cosine_similarities[i])
        for i in range(len(recipies))
    ]
    results.sort(key=lambda x: x[2], reverse=True)

    if threshold is not None:
        results = [r for r in results if r[2] >= threshold]

    if k is not None:
        results = results[:k]
    return results


stop_words = set(stopwords.words("english"))
# remove words which are meaningless in context of cooking
stop_words.update(
    [
        "add",
        "added",
        "adding",
        "addition",
        "also",
        "almost",
        "another",
        "easily",
        "easy",
    ]
)
lemmatizer = WordNetLemmatizer()


def preprocess_text(doc):
    doc = doc.translate(str.maketrans("", "", string.punctuation)).lower()

    words = word_tokenize(doc)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and word.isalpha()
    ]

    return " ".join(words)
