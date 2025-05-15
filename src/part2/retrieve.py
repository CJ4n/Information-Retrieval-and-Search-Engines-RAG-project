from sklearn.metrics.pairwise import cosine_similarity


def retrieve_documents(
    query_embeddings, recipe_embeddings, recipe_texts, recipe_ids, k, threshold
):
    if len(recipe_texts) != len(recipe_ids):
        raise ValueError("Recipes and recipe_ids must have the same length")
    if k is None and threshold is None:
        raise ValueError("Either k or threshold must be specified")

    cosine_similarities = cosine_similarity(
        query_embeddings, recipe_embeddings
    ).flatten()

    results = [
        (recipe_texts[i], recipe_ids[i], cosine_similarities[i])
        for i in range(len(recipe_texts))
    ]
    results.sort(key=lambda x: x[2], reverse=True)

    if threshold is not None:
        results = [r for r in results if r[2] >= threshold]

    if k is not None:
        results = results[:k]
    return results
