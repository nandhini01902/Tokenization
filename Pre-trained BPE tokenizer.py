from transformers import AutoTokenizer

if __name__ == "__main__":

    corpus = [
        "low lower lowest",
        "low low lower",
        "newer newer newest",
        "token tokenization",
        "I love learning AI",
    ]

    

   
    pretrained_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    
    examples = [
        "low lower",
        "lowest!",
        "I love learning AI",
        "Tokenization is amazing!",
        "Tokens to token id",
    ]

    for text in examples:
        print("\nInput:", repr(text))

        pretrained_ids = pretrained_tokenizer.encode(
            text, add_special_tokens=False
        )
        print("GPT-2 Token IDs:  ", pretrained_ids)