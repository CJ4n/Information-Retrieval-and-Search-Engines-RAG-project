import json
import os

import datasets
import pandas as pd

# from google.colab import userdata
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from evaluation import *
from metrics import *
from prompts import *
from retrieve import *
from retrieve import retrieve_documents
from utils import *


def print_query(
    query_text,
    recipies,
    recipe_ids,
    best_K,
    best_threshold,
    df,
    vec_uni,
    vec_bi,
    X_uni,
    X_bi,
):
    results = retrieve_documents(
        query_text=query_text,
        recipies=recipies,
        recipe_ids=recipe_ids,
        k=best_K,
        threshold=best_threshold,
        vec_uni=vec_uni,
        vec_bi=vec_bi,
        X_uni=X_uni,
        X_bi=X_bi,
    )
    output_dir = "tfidf_problems"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{query_text}.txt")

    with open(output_file, "w") as file:
        file.write(f"Query: {query_text}\n")
        for recipe, recipe_id, score in results:
            file.write(f"Recipe ID: {recipe_id}, Score: {score:.4f}\n")
            recipe_info = df[df["official_id"] == recipe_id].iloc[0]
            file.write(f"Name: {recipe_info['name']}\n")
            file.write(f"Description: {recipe_info['description']}\n")
            file.write(f"Ingredients: {recipe_info['ingredients']}\n")
            file.write(f"Steps: {recipe_info['steps']}\n\n")


def load_data():
    print("Loading data")
    dataset = datasets.load_dataset(
        "parquet", data_files="./irse_documents_2025_recipes.parquet"
    )["train"]

    df = dataset.to_pandas()

    queries_data = json.load(open("./irse_queries_2025_recipes.json", "r"))

    print("Number of loaded documents:", len(df))
    print("Number of loaded queries:", len(queries_data["queries"]))
    return df, queries_data


def preprocess_data(
    df, queries_data, fields_to_include=["name", "description", "ingredients", "steps"]
):
    print("Preprocessing data")
    recipies = df.apply(
        lambda row: " ".join([row[field] for field in fields_to_include]),
        axis=1,
    )

    recipe_ids = df["official_id"]
    print("Number of preprocessed documents:", len(recipies))

    queries = pd.DataFrame(columns=["q", "r", "a"])
    for query_item in queries_data["queries"]:
        # print("Processing query:", query_item)
        query_text = query_item["q"]
        relevance_pairs = query_item["r"]
        answer = query_item["a"]
        queries = pd.concat(
            [
                queries,
                pd.DataFrame(
                    {"q": [query_text], "r": [relevance_pairs], "a": [answer]}
                ),
            ],
            ignore_index=True,
        )

    print("Number of preprocessed queries:", len(queries))
    return recipies, recipe_ids, queries


def preprocess_recipes(recipies):
    # TODO commnet this otu
    print("Preprocessing recipes")
    # filename = "preprocessed_recipes6.pkl"
    # if os.path.exists(filename):
    #     with open(filename, "rb") as f:
    #         preprocessed_recipes = pickle.load(f)
    # else:
    #     preprocessed_recipes = [preprocess_text(doc) for doc in tqdm(recipies)]
    #     with open(filename, "wb") as f:
    #         pickle.dump(preprocessed_recipes, f)  # Adjust based on the format used
    preprocessed_recipes = [preprocess_text(doc) for doc in tqdm(recipies)]
    return preprocessed_recipes


def get_vectorizers(preprocessed_recipes):
    # print("Loading TF-IDF vectorizers")
    # filename_uni = "vectorizer_uni.pkl"
    # filename_bi = "vectorizer_bi.pkl"
    # filename_X_uni = "X_uni.pkl"
    # filename_X_bi = "X_bi.pkl"

    # if (
    #     os.path.exists(filename_uni)
    #     and os.path.exists(filename_bi)
    #     and os.path.exists(filename_X_uni)
    #     and os.path.exists(filename_X_bi)
    # ):
    #     with open(filename_uni, "rb") as f:
    #         vec_uni = pickle.load(f)
    #     with open(filename_bi, "rb") as f:
    #         vec_bi = pickle.load(f)
    #     with open(filename_X_uni, "rb") as f:
    #         X_uni = pickle.load(f)
    #     with open(filename_X_bi, "rb") as f:
    #         X_bi = pickle.load(f)
    # else:
    #     print("Fitting TF-IDF vectorizers")
    #     vec_uni = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1, 1))
    #     vec_bi = TfidfVectorizer(
    #         min_df=50,
    #         max_df=0.4,
    #         ngram_range=(2, 2),
    #         max_features=10000,
    #     )

    #     X_uni = vec_uni.fit_transform(preprocessed_recipes)
    #     X_bi = vec_bi.fit_transform(preprocessed_recipes)

    #     with open(filename_uni, "wb") as f:
    #         pickle.dump(vec_uni, f)
    #     with open(filename_bi, "wb") as f:
    #         pickle.dump(vec_bi, f)
    #     with open(filename_X_uni, "wb") as f:
    #         pickle.dump(X_uni, f)
    #     with open(filename_X_bi, "wb") as f:
    #         pickle.dump(X_bi, f)

    print("Fitting TF-IDF vectorizers")
    vec_uni = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1, 1))
    vec_bi = TfidfVectorizer(
        min_df=50,
        max_df=0.4,
        ngram_range=(2, 2),
        max_features=10000,
    )

    X_uni = vec_uni.fit_transform(preprocessed_recipes)
    X_bi = vec_bi.fit_transform(preprocessed_recipes)

    return vec_uni, vec_bi, X_uni, X_bi
