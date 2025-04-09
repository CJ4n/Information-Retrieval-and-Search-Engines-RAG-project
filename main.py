# from google.colab import userdata
# userdata.get("HF_TOKEN")
import json
import os
import string

import datasets
import dotenv
import nltk
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# HF_TOKEN = os.getenv("HF_TOKEN")

# if HF_TOKEN:
#     print("Hugging Face token loaded successfully!")
# else:
#     print("Warning: HF_TOKEN not found in environment variables.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
device = "cpu"


# TASKS:
#
# 1. find TF IDF impl.
# 2. define term vocabulary -> preprocessing (fleksja, stop words -> look at excercise)
# - setup validation pipeline
# 3. support for multi word expression (just preprocessing???)
# 4. which parts of docs. to embedd + run some experiment to decide and validate
# 5. impl. of find kNN search (mb. just numpy is enough)
#    - what similairty
#    - statistics (F1, recall, ect (6 in total!!), MAP.)
# 6. setup integarion w/ LLM
# 7. try out different prompts (at laest 2)
# 8. qualitative anallysi -> can LLM reason accross respcei, etc etc (read task)
lemmatizer = WordNetLemmatizer()


def preprocess(doc):  # this takes in a string and converts it into a list
    doc = doc.split()
    preprocessed_text = []
    for text in doc:
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )  # remove punctuation
        text = text.lower()  # convert to lower case
        words = word_tokenize(text)  # tokenize the text
        words = [
            word for word in words if word not in stopwords.words("english")
        ]  # remove stopwords

        words = [lemmatizer.lemmatize(word) for word in words]  # lemmatize
        if words != []:
            preprocessed_text.append(words[0])
    return preprocessed_text


def vectorize_text():
    vectorizer = TfidfVectorizer(tokenizer=preprocess)
    X = vectorizer.fit_transform(data["text"])  # this will take some time to run
    tfidf_df = pd.DataFrame(
        X.toarray(), index=range(len(data)), columns=vectorizer.get_feature_names_out()
    )
    tfidf_df = tfidf_df.round(2)
    tfidf_df


def load_data():
    try:
        recipes = datasets.load_dataset(
            "parquet", data_files="./irse_documents_2025_recipes.parquet"
        )["train"]
    except (FileNotFoundError, ValueError):
        print("Parquet file not found. Downloading...")
        os.system(
            "wget https://people.cs.kuleuven.be/~thomas.bauwens/irse_documents_2025_recipes.parquet"
        )
        recipes = datasets.load_dataset(
            "parquet", data_files="./irse_documents_2025_recipes.parquet"
        )["train"]

    try:
        queries = json.load(open("./irse_queries_2025_recipes.json", "r"))
    except FileNotFoundError:
        print("JSON file not found. Downloading...")
        os.system(
            "wget https://people.cs.kuleuven.be/~thomas.bauwens/irse_queries_2025_recipes.json"
        )
        queries = json.load(open("./irse_queries_2025_recipes.json", "r"))
    return recipes, queries


sample_queries = [
    "a cajun style gumbo with an easy roux",
    "I am feeling like eating shrimp tacos tonight. What's a good recipe?",
    "recipe for easy vegetarian lasagna",
    "How do I make spageti and meatballs?",
    "15 minute lunch recipe",
    "Give me suggestion for some easy vegetarian weeknight dinner recipes",
]
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


def get_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model


def main():
    recipes, queries = load_data()
    # TODO: implement TF-IDF
    # TODO: fit TF-IDF model on recipes dataset
    # TODO: implement nearest neighbors search
    prompt = """

    YOUR PROMPT GOES HERE

    """
    model = get_llm()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_string = prompt + sample_queries[0]
    encoded_prompt = tokenizer(
        input_string, return_tensors="pt", add_special_tokens=False
    )

    # encoded_prompt = encoded_prompt.to(device)
    generated_ids = model.generate(
        **encoded_prompt, max_new_tokens=1000, do_sample=True
    )
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])

