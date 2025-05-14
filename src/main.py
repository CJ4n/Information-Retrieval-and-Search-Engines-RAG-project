DEBUG = False
from functools import partial
from multiprocessing import Pool

import nltk

# from google.colab import userdata
from huggingface_hub import login
from tqdm import tqdm

from evaluation import *
from llm import *
from metrics import *
from prompts import *
from retrieve import *
from utils import *

tqdm.pandas()
login(token="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")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")
DEBUG = False


# vocab_uni = vec_uni.get_feature_names_out()
# idf_values_uni = vec_uni.idf_

# sorted_terms_uni = sorted(
#     list(zip(vocab_uni, idf_values_uni)), key=lambda x: x[1], reverse=True
# )

# sorted_terms_uni


# vocab_bi = vec_bi.get_feature_names_out()
# idf_values_bi = vec_bi.idf_

# sorted_terms_bi = sorted(
#     list(zip(vocab_bi, idf_values_bi)), key=lambda x: x[1], reverse=True
# )
# sorted_terms_bi


# chosen using grid search over hyper params space


# retrieve_documents(
#     query_text="",
#     recipies=recipies,
#     recipe_ids=recipe_ids,
#     k=best_K,
#     threshold=None,
#     vec_uni=vec_uni,
#     vec_bi=vec_bi,
#     X_uni=X_uni,
#     X_bi=X_bi,
# )
# metrics = evaluate_ir_system(
#     queries=queries,
#     recipies=recipies,
#     recipe_ids=recipe_ids,
#     k=best_K,
#     threshold=best_threshold,
#     vec_uni=vec_uni,
#     vec_bi=vec_bi,
#     X_uni=X_uni,
#     X_bi=X_bi,
# )

# metrics


# test_queries = [
#     "Where can I follow cooking classes",  # should not return anything, but retursn -> it's very hard to cut off noisy data using thresholds, as sometimes irrelvant documents have hihger score, then relevant ones for different queries ->
#     "How does Gordon Ramsay make his beef Wellington?",  # found things wiht gordron ramsey or beef wellington, but nothing with noth -> ignores context
#     "Do you know any soups from Paraguay?",  # "Paragya" appears in only two documents, so it's extremally rare, since I require term to appear in more then 20 documents, it does not appear in TF_IDF so it's ignored -> TF-IDF ignores very rare terms
#     "How do you make piza",  # does not handle types -> problem with 'synonyms'
#     "I do not want to eat pizza, what can I eat instead?",  # -> did not capture negation
# ]


# print_query(test_queries[3], recipies, recipe_ids, best_K, best_threshold)

# query = "I want to eat something with cactus. How many recipes do you know?"


def evaluate_combination(combination, df, queries_data, K, threshold):
    recipies, recipe_ids, queries = preprocess_data(df, queries_data, combination)
    recipies = preprocess_recipes(recipies)
    vec_uni, vec_bi, X_uni, X_bi = get_vectorizers(recipies)
    metrics = evaluate_ir_system(
        queries, recipies, recipe_ids, K, threshold, vec_uni, vec_bi, X_uni, X_bi
    )
    return combination, metrics


def experiment_which_fields_are_important():
    print("#########################")
    print("Running experiment to find most important fields")
    print("#########################")

    df, queries_data = load_data()
    K = 40
    threshold = 0.2
    combinations = [
        ["name", "description", "ingredients", "steps"],
        ["description", "ingredients", "steps"],
        ["description", "ingredients"],
        ["description", "steps"],
        ["description"],
    ]

    func = partial(
        evaluate_combination, df=df, queries_data=queries_data, K=K, threshold=threshold
    )

    with Pool(5) as pool:
        results = list(
            tqdm(pool.map(func, combinations), desc="Evaluating combinations")
        )

    for combination, metrics in results:
        print(f"Metrics for {combination}: {metrics}")
        print(f"F1: {metrics['macro_f1']}")


def experiment_run_all_prompt_generation():
    print("#########################")
    print("Running all prompt generation experiments")
    print("#########################")

    prompts = {
        "propt_cactus_bad_prompt": propt_cactus_bad_prompt,
        "propmt_borscht_reasoning": propmt_borscht_reasoning,
        "prompt_cactus_reasoning": prompt_cactus_reasoning,
        "prompt_calamari_reasoning": prompt_calamari_reasoning,
        "prompt_rhode_reasoning": prompt_rhode_reasoning,
        "prompt_arab_dish_reasoning": prompt_arab_dish_reasoning,
        "prompt_ramsy_resoning": prompt_ramsy_resoning,
        "prompt_with_scores": prompt_with_scores,
        "prompt_without_scores": prompt_without_scores,
        "prompot_wihtout_documents": prompot_wihtout_documents,
    }

    for prompt_name, prompt in prompts.items():
        response = generate_response(prompt)
        print(response)
        with open(f"prompts/{prompt_name}.txt", "w") as f:
            f.write(prompt)


def experiment_find_best_hyperparams():
    print("#########################")
    print("Running experiment to find best hyperparams")
    print("#########################")

    df, queries_data = load_data()
    recipies, recipe_ids, queries = preprocess_data(df, queries_data)
    recipies = preprocess_recipes(recipies)

    vec_uni, vec_bi, X_uni, X_bi = get_vectorizers(recipies)
    results = create_parameter_heatmap(
        queries, recipies, recipe_ids, vec_uni, vec_bi, X_uni, X_bi
    )
    print(f"Best hyperparams: {results}")


def experiment_calculate_metrics():
    print("#########################")
    print("Running experiment to calculate metrics")
    print("#########################")

    df, queries_data = load_data()
    best_K = 40
    best_threshold = 0.2
    recipies, recipe_ids, queries = preprocess_data(df, queries_data)
    recipies = preprocess_recipes(recipies)
    vec_uni, vec_bi, X_uni, X_bi = get_vectorizers(recipies)
    metrics = evaluate_ir_system(
        queries=queries,
        recipies=recipies,
        recipe_ids=recipe_ids,
        k=best_K,
        threshold=best_threshold,
        vec_uni=vec_uni,
        vec_bi=vec_bi,
        X_uni=X_uni,
        X_bi=X_bi,
    )
    print(metrics)


def experiment_drawbacks_of_tfidf():
    print("#########################")
    print("Running experiment to find drawbacks of TF-IDF")
    print("#########################")

    df, queries_data = load_data()
    best_K = 40
    best_threshold = 0.2
    recipies, recipe_ids, queries = preprocess_data(df, queries_data)
    recipies = preprocess_recipes(recipies)
    vec_uni, vec_bi, X_uni, X_bi = get_vectorizers(recipies)
    test_queries = [
        "Where can I follow cooking classes",  # should not return anything, but retursn -> it's very hard to cut off noisy data using thresholds, as sometimes irrelvant documents have hihger score, then relevant ones for different queries ->
        "How does Gordon Ramsay make his beef Wellington?",  # found things wiht gordron ramsey or beef wellington, but nothing with noth -> ignores context
        "Do you know any soups from Paraguay?",  # "Paragya" appears in only two documents, so it's extremally rare, since I require term to appear in more then 20 documents, it does not appear in TF_IDF so it's ignored -> TF-IDF ignores very rare terms
        "How do you make piza",  # does not handle types -> problem with 'synonyms'
        "I do not want to eat pizza, what can I eat instead?",  # -> did not capture negation
    ]

    for i, query in enumerate(test_queries):
        print(f"{i}. {query}")
        print_query(
            query_text=query,
            recipies=recipies,
            recipe_ids=recipe_ids,
            df=df,
            best_K=best_K,
            best_threshold=best_threshold,
            vec_uni=vec_uni,
            vec_bi=vec_bi,
            X_uni=X_uni,
            X_bi=X_bi,
        )
        print("-" * 100)


if __name__ == "__main__":
    # experiment_which_fields_are_important()
    # experiment_find_best_hyperparams()
    # experiment_calculate_metrics()
    # experiment_drawbacks_of_tfidf()
    experiment_run_all_prompt_generation()
