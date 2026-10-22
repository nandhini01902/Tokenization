class WhitespaceTokenizer:
    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}

    def tokenize(self, text):
        return text.split()

    def assign(self, texts):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    token_id = len(self.vocab)
                    self.vocab[token] = token_id
                    self.id_to_token[token_id] = token

    def encode(self, text):
        return [self.vocab[token] for token in self.tokenize(text)]

    def decode(self, token_ids):
        return " ".join(
            self.id_to_token[token_id] for token_id in token_ids
        )

texts = [
    "I love NLP",
    "I love tokenization!",
    "Tokenization is useful"
]

tokenizer = WhitespaceTokenizer()
tokenizer.assign(texts)

sentence = "I love NLP"
tokens = tokenizer.tokenize(sentence)
ids = tokenizer.encode(sentence)

print("Tokens:", tokens)
print("Token IDs:", ids)
print("Decoded:", tokenizer.decode(ids))
print("Vocabulary:", tokenizer.vocab)
