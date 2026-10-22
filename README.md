# Tokenizer Implementation and Comparison

The project compares these implementations with pretrained tokenizers from Hugging Face Transformers:

- **BERT (`bert-base-uncased`)** — uses WordPiece tokenization.
- **GPT-2 (`gpt2`)** — uses byte-level Byte Pair Encoding (BPE).

The goal is to understand tokenization, vocabulary construction, token ID assignment, subword segmentation, and the differences between educational implementations and pretrained tokenizers.

## Objectives

- Understand how text is converted into tokens and token IDs.
- Implement whitespace tokenization without a pretrained model.
- Implement a basic BPE algorithm without machine learning libraries.
- Experiment with pretrained BERT and GPT-2 tokenizers.
- Compare token boundaries, token counts, and token IDs.
- Identify limitations such as unknown-token handling and lossy decoding.

## Structure

The project contains the following Python files:

| File | Purpose |
|---|---|
| `Whitespace.py` | Custom whitespace tokenizer |
| `Pre-trained Whitespace tokenizer.py` | Pretrained BERT tokenizer comparison |
| `bpe_tokenizer.py` | Custom educational BPE implementation |
| `Pre-trained BPE tokenizer.py` | Pretrained GPT-2 tokenizer comparison |

*Note: The pretrained whitespace script uses BERT WordPiece, not whitespace-only tokenization. Renaming it to `pretrained_bert_tokenizer.py` would make its purpose clearer.*


# Whitespace Tokenizer Comparison

###  Custom Implementation

**File:** `Whitespace.py`

The custom whitespace tokenizer uses Python's `text.split()` method to separate text wherever whitespace occurs.

Its main operations are:

- `tokenize(text)` — splits text into tokens.
- `assign(texts)` — builds a vocabulary and assigns an integer ID to each new token.
- `encode(text)` — converts tokens into IDs.
- `decode(token_ids)` — converts IDs back into tokens and joins them with spaces.

Token IDs are assigned in the order that unique tokens first appear in the training texts.

#### Example results

| Input | Tokens | Token IDs |
|---|---|---|
| `I love NLP` | `I`, `love`, `NLP` | `[0, 1, 2]` |
| `I love tokenization!` | `I`, `love`, `tokenization!` | `[0, 1, 3]` |
| `Tokenization is useful` | `Tokenization`, `is`, `useful` | `[4, 5, 6]` |

#### Vocabulary

```python
{
    "I": 0,
    "love": 1,
    "NLP": 2,
    "tokenization!": 3,
    "Tokenization": 4,
    "is": 5,
    "useful": 6
}
```

#### Observations

- Punctuation remains attached to words. For example, `tokenization!` is one token.
- Capitalization is preserved, so `Tokenization` and `tokenization!` are different tokens.
- Each known word receives one ID.
- Encoding a token that is absent from the vocabulary raises a `KeyError` in this implementation.

This tokenizer is simple and easy to understand, but it does not handle unseen words automatically.

###  Pretrained BERT Tokenizer

**File:** `Pre-trained Whitespace tokenizer.py`

This script loads the pretrained `bert-base-uncased` tokenizer from Hugging Face Transformers.

Despite the original filename, this is a **WordPiece tokenizer**, not a whitespace-only tokenizer.

BERT lowercases text and can split a word into smaller subword units. The script uses `add_special_tokens=False`, so the displayed IDs exclude BERT's `[CLS]` and `[SEP]` special tokens.

#### Example results

| Input | BERT tokens | Token IDs |
|---|---|---|
| `I love NLP` | `i`, `love`, `nl`, `##p` | `[1045, 2293, 17953, 2361]` |
| `I love tokenization!` | `i`, `love`, `token`, `##ization`, `!` | `[1045, 2293, 19204, 3989, 999]` |
| `Tokenization is useful` | `token`, `##ization`, `is`, `useful` | `[19204, 3989, 2003, 6179]` |

#### Observations

- BERT lowercases the input.
- A word can become multiple subword tokens.
- Punctuation can be represented by a separate token.
- IDs come from BERT's pretrained vocabulary rather than a vocabulary built from the three example sentences.

Unlike the custom whitespace tokenizer, BERT can represent many words using familiar subword units, including words that were not present as complete words in the example inputs.

##  BPE Tokenizer Comparison

###  Custom BPE Implementation

**File:** `bpe_tokenizer.py`

The custom BPE tokenizer is an educational implementation written using Python's `Counter` from the standard library. It does not use a pretrained model or machine learning framework.

The implementation follows these steps:

1. Split each training text into individual characters, including spaces and punctuation.
2. Count adjacent symbol pairs across the training corpus.
3. Select the most frequent pair.
4. Merge the selected pair wherever it occurs as an adjacent pair.
5. Repeat while the vocabulary is below the configured limit and the best pair occurs at least twice.
6. Assign IDs to the initial symbols and newly learned merged symbols.
7. Use `<UNK>` for resulting symbols that are not in the vocabulary.

The implementation uses deterministic tie-breaking: when pair frequencies are equal, pairs are ordered lexicographically.

#### Example results

| Input | Custom BPE tokens | Token IDs |
|---|---|---|
| `low lower` | `low`, ` lower` | `[18, 24]` |
| `lowest!` | `low`, `est`, `<UNK>` | `[18, 28, 32]` |
| `I love learning AI` | `I`, space, `lo`, `v`, `e`, space, `l`, `e`, `a`, `r`, `n`, `i`, `n`, `g`, space, `A`, `I` | `[2, 0, 17, 14, 4, 0, 8, 4, 3, 11, 9, 6, 9, 5, 0, 1, 2]` |
| `Tokenization is amazing!` | `<UNK>`, `oken`, `i`, `z`, `a`, `t`, `i`, `o`, `n`, space, `i`, `s`, space, `a`, `<UNK>`, `a`, `z`, `i`, `n`, `g`, `<UNK>` | `[32, 30, 6, 16, 3, 13, 6, 10, 9, 0, 6, 12, 0, 3, 32, 3, 16, 6, 9, 5, 32]` |
| `Tokens to token id` | `<UNK>`, `oken`, `s`, space, `t`, `o`, space, `token`, space, `i`, `<UNK>` | `[32, 30, 12, 0, 13, 10, 0, 31, 0, 6, 32]` |


#### Observations and limitations

- The custom tokenizer learns merges from the small training corpus.
- Its vocabulary and token IDs are specific to this implementation and training data.
- It is case-sensitive.
- Characters absent from the training corpus may become `<UNK>`.
- The decoder removes `<UNK>` rather than restoring the original unknown character, so decoding can lose information.
- Because spaces are ordinary symbols, merges can include spaces and can cross word boundaries.

This implementation demonstrates the basic idea behind BPE, but it is not a complete reproduction of production BPE tokenizers.

###  Pretrained GPT-2 BPE Tokenizer

**File:** `Pre-trained BPE tokenizer.py`

This script loads the pretrained GPT-2 tokenizer from Hugging Face Transformers.

GPT-2 uses **byte-level BPE**. Its vocabulary and merge rules were learned before the comparison script runs; the local training corpus is not used to train GPT-2.

#### Example results

| Input | GPT-2 tokens | Token IDs |
|---|---|---|
| `low lower` | `low`, `Ġlower` | `[9319, 2793]` |
| `lowest!` | `low`, `est`, `!` | `[9319, 395, 0]` |
| `I love learning AI` | `I`, `Ġlove`, `Ġlearning`, `ĠAI` | `[40, 1842, 4673, 9552]` |
| `Tokenization is amazing!` | `Token`, `ization`, `Ġis`, `Ġamazing`, `!` | `[30642, 1634, 318, 4998, 0]` |
| `Tokens to token id` | `Tokens`, `Ġto`, `Ġtoken`, `Ġid` | `[22906, 284, 11241, 4686]` |

Here, `Ġ` is the visible marker used by GPT-2's tokenizer to indicate a leading space in a token representation.

*Verify these IDs against the exact `gpt2` tokenizer version used by your script before submitting the report.*

#### Observations

- GPT-2 uses a pretrained vocabulary and merge rules.
- A word can be represented by one token or several subword tokens.
- Leading spaces can affect token boundaries.
- Byte-level encoding allows it to represent text beyond the characters present in the small custom training corpus.
- Its token IDs belong to GPT-2's vocabulary, not the custom BPE vocabulary.

##  Overall Comparison

| Feature | Custom Whitespace | Pretrained BERT | Custom BPE | Pretrained GPT-2 |
|---|---|---|---|---|
| Tokenization method | Whitespace splitting | WordPiece | Character-pair merging | Byte-level BPE |
| Vocabulary source | Example training texts | Pretrained vocabulary | Example training corpus | Pretrained vocabulary |
| Subword splitting | No | Yes | Yes, through learned merges | Yes |
| Requires pretrained tokenizer files | No | Yes | No | Yes |
| Unknown input handling | May raise `KeyError` | Uses pretrained vocabulary and tokenization rules | Uses `<UNK>` for unsupported resulting symbols | Byte-level encoding handles arbitrary text |
| Token IDs | Custom IDs | BERT IDs | Custom IDs | GPT-2 IDs |

### Key findings

1. **Whitespace tokenization** is easy to implement, but punctuation and capitalization can create separate vocabulary entries.
2. **WordPiece tokenization** allows BERT to break words into subword units from its pretrained vocabulary.
3. **Custom BPE** demonstrates how frequent adjacent symbols can be merged to build a vocabulary.
4. **GPT-2 BPE** uses pretrained byte-level rules and is more robust to unseen text than this educational character-based implementation.
5. **Token IDs are vocabulary-specific.** The same numerical ID can represent completely different tokens in different tokenizers.

## Testing and Validation

The following checks are useful when testing the project:

- Confirm that each script runs without errors.
- Check the tokens and IDs for every sample input.
- Verify that decoding restores the original input where the tokenizer supports lossless reconstruction.
- Test punctuation, capitalization, repeated words, whitespace, and unseen characters.
- Compare token counts across the custom and pretrained implementations.
- Verify pretrained tokenizer results using the exact model identifiers used in the scripts.

The sample outputs in this README should match the output generated by the final versions of the Python files.

##  Conclusion

This project demonstrates how tokenization converts text into smaller units and maps those units to numerical IDs.

Implementing whitespace tokenization and BPE from scratch provides a practical understanding of vocabulary construction, token boundaries, pair merging, and encoding. Comparing these implementations with pretrained BERT and GPT-2 tokenizers highlights how production tokenizers use pretrained vocabularies and more advanced segmentation strategies.

The project provides a foundation for further exploration of tokenization, subword algorithms, vocabulary design, and the preprocessing steps used by large language models.

