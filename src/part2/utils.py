import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer


def loadWikirQueries(wikir_path: Path, split: str) -> Dataset:
    split_path = wikir_path / split
    if not split_path.is_dir():
        raise ValueError(f"Split {split} not found in {wikir_path}.")

    queries = pd.read_csv(split_path / "queries.csv")
    qrels = pd.read_csv(split_path / "qrels", sep="\t", header=None)
    qrels.columns = ["id_left", "number", "id_right", "relevance"]
    qrels = qrels.merge(queries, on="id_left")
    qrels = qrels.rename(
        columns={"id_left": "query_id", "id_right": "doc_id", "text_left": "query"}
    )
    qrels = qrels.drop(columns=["number", "query_id"])

    return Dataset.from_pandas(qrels, preserve_index=False)


def loadWikir(wikir_path: Path) -> Tuple[Dataset, DatasetDict]:
    queries_train = loadWikirQueries(wikir_path, "training")
    queries_valid = loadWikirQueries(wikir_path, "validation")
    queries_test = loadWikirQueries(wikir_path, "test")

    documents = pd.read_csv(wikir_path / "documents.csv")
    documents = documents.rename(
        columns={"id_right": "doc_id", "text_right": "doc_text"}
    )
    return Dataset.from_pandas(documents), DatasetDict(
        {"train": queries_train, "validation": queries_valid, "test": queries_test}
    )


def queryDatasetToQueryJson(queries: Dataset) -> dict:
    queries_to_documents = defaultdict(list)
    for example in queries:
        q = example["query"]
        d = example["doc_id"]
        r = example["relevance"]
        queries_to_documents[q].append([d, r])

    return {
        "queries": [
            {"q": query, "r": documents}
            for query, documents in queries_to_documents.items()
        ]
    }


def get_model_and_tokenizer():
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    cpu_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    tokenizer = model.tokenizer
    return model, cpu_model, tokenizer


def check_tokenization(text, tokenizer):
    token_ids = tokenizer.encode(text)

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Original text: '{text}'")
    print(f"Tokenized as: {tokens}")


def embed_documents_part_2(model, documents):
    if os.path.exists("wiki_embeddings.npy"):
        wiki_embeddings = np.load("wiki_embeddings.npy")
    else:
        wiki_embeddings = model.encode(
            documents["doc_text"],
            batch_size=500,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            num_workers=0,
        )
        np.save("wiki_embeddings.npy", wiki_embeddings)
    return wiki_embeddings


def embed_documents_part_1(model, documents):
    if os.path.exists("part1_embeddings.npy"):
        embeddings = np.load("part1_embeddings.npy")
    else:
        embeddings = model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save("part1_embeddings.npy", embeddings)
    return embeddings


def get_queries_data(queries_data, cpu_model):
    queries = pd.DataFrame(columns=["q", "r"])
    for query_item in queries_data["queries"]:
        query_text = query_item["q"]
        relevance_pairs = query_item["r"]
        queries = pd.concat(
            [
                queries,
                pd.DataFrame({"q": [query_text], "r": [relevance_pairs]}),
            ],
            ignore_index=True,
        )

    embeddings = cpu_model.encode(
        queries["q"],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        num_workers=0,
    )

    queries["embeddings"] = list(embeddings)
    return queries


def load_data_part2(cpu_model):
    data_path = Path("wikIR1k")
    documents, queries = loadWikir(data_path)
    test_queries = queryDatasetToQueryJson(queries["test"])
    train_queries = queryDatasetToQueryJson(queries["train"])
    validation_queries = queryDatasetToQueryJson(queries["validation"])
    train_queries = get_queries_data(train_queries, cpu_model)
    validation_queries = get_queries_data(validation_queries, cpu_model)
    test_queries = get_queries_data(test_queries, cpu_model)
    print("Train queries: ", len(train_queries["q"]))
    print("Validation queries: ", len(validation_queries["q"]))
    print("Test queries: ", len(test_queries["q"]))
    document_id_to_idx = {d["doc_id"]: idx for idx, d in enumerate(documents)}
    return (
        documents,
        test_queries,
        train_queries,
        validation_queries,
        document_id_to_idx,
    )


def get_documents_data_part_1():
    cooking_dataset = datasets.load_dataset(
        "parquet", data_files="./irse_documents_2025_recipes.parquet"
    )["train"]

    df = cooking_dataset.to_pandas()

    recipies_cooking = df.apply(
        lambda row: f"{row['name']} {row['description']} {row['ingredients']} {row['steps']}",
        axis=1,
    )

    recipe_ids = cooking_dataset["official_id"]
    return recipies_cooking, recipe_ids


def load_data_part_1(cpu_model):
    queries_data = json.load(open("./irse_queries_2025_recipes.json", "r"))
    queries = get_queries_data(queries_data, cpu_model)
    recipies_cooking, recipe_ids = get_documents_data_part_1()
    return queries, recipies_cooking, recipe_ids
