from collections import Counter


class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.unk_token = "<UNK>"

    def _split(self, text):
        return list(text)

    def _count_pairs(self, sequences):
        counts = Counter()

        for symbols in sequences:
            for i in range(len(symbols) - 1):
                counts[(symbols[i], symbols[i + 1])] += 1

        return counts

    def _merge_pair(self, symbols, pair):
        merged = []
        i = 0

        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and (symbols[i], symbols[i + 1]) == pair
            ):
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1

        return merged

    def train(self, texts):
        sequences = [self._split(text) for text in texts]

        base_symbols = sorted(
            {symbol for seq in sequences for symbol in seq}
        )

        self.merges = []
        self.vocab = {symbol: i for i, symbol in
                      enumerate(base_symbols)}

        while len(self.vocab) < self.vocab_size:
            pair_counts = self._count_pairs(sequences)

            if not pair_counts:
                break

            
            best_pair, frequency = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0])
            )

            if frequency < 2:
                break

            merged_symbol = best_pair[0] + best_pair[1]

            self.merges.append(best_pair)

            if merged_symbol not in self.vocab:
                self.vocab[merged_symbol] = len(self.vocab)

            sequences = [
                self._merge_pair(seq, best_pair)
                for seq in sequences
            ]

        self.unk_id = len(self.vocab)
        self.vocab[self.unk_token] = self.unk_id
        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

        return self

    def tokenize(self, text):
        symbols = self._split(text)

        for pair in self.merges:
            symbols = self._merge_pair(symbols, pair)

        return [
            symbol if symbol in self.vocab else self.unk_token
            for symbol in symbols
        ]

    def encode(self, text):
        return [
            self.vocab.get(token, self.unk_id)
            for token in self.tokenize(text)
        ]

    def decode(self, token_ids):
        pieces = [
            self.id_to_token.get(i, self.unk_token)
            for i in token_ids
        ]
        return "".join(
            piece for piece in pieces
            if piece != self.unk_token
        )


if __name__ == "__main__":
    corpus = [
        "low lower lowest",
        "low low lower",
        "newer newer newest",
        "token tokenization",
        "I love learning AI",
    ]

    tokenizer = BPETokenizer(vocab_size=80)
    tokenizer.train(corpus)
    examples = [
        "low lower",
        "lowest!",
        "I love learning AI",
        "Tokenization is amazing!",
        "Tokens to token id",
    ]

    
    for text in examples:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)

        print("\nText: ", repr(text))
        print("Tokens:", tokens)
        print("IDs:   ", ids)
        print("Count: ", len(tokens))
        print("Decoded:", repr(tokenizer.decode(ids)))
