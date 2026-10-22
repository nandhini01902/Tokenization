from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "I love NLP",
    "I love tokenization!",
    "Tokenization is useful"
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    print("\nInput:", text)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)