import torch
import transformers
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import *
from retrieve import retrieve_documents

login(token="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")
DEVICE = "cpu"

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model, tokenizer = None, None


def get_model_and_tokenizer():
    global model, tokenizer
    if model is None or tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def generate_response(prompt):
    model, tokenizer = get_model_and_tokenizer()

    encoded_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt"
    )
    encoded_prompt = encoded_prompt.to(DEVICE)
    generated_ids = model.generate(encoded_prompt, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]


def prepare_prompt(
    query, prompt_template, df, recipies, recipe_ids, k=5, threshold=None
):
    results = retrieve_documents(query, recipies, recipe_ids, k=k, threshold=threshold)
    retrieved_recipes = ""

    for idx, (recipe, recipe_id, score) in enumerate(results):
        recipe_info = df[df["official_id"] == recipe_id].iloc[0]
        retrieved_recipes += f"Document {idx}, Score: {score:.4f}\n"
        retrieved_recipes += f"Name: {recipe_info['name']}\n"
        retrieved_recipes += f"Description: {recipe_info['description']}\n"
        retrieved_recipes += f"Ingredients: {recipe_info['ingredients']}\n"
        retrieved_recipes += f"Steps: {recipe_info['steps']}\n\n"

        print(f"Document {idx}")
        print(f"Name: {recipe_info['name']}")
        print(f"Description: {recipe_info['description']}")
        print(f"Ingredients: {recipe_info['ingredients']}")
        print(f"Steps: {recipe_info['steps']}\n")
        print(f"Score: {score:.4f}\n")

    prompt = prompt_template.format(
        retrieved_recipes=retrieved_recipes, user_query=query
    )
    return prompt
