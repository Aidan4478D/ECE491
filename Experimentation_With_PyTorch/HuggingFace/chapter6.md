## Hugging Face Chapter 5 Notes

## Module 5 - The Tokenizers Library

- [Training a New Tokenizer from Old](#training-a-new-tokenizer-from-old)
- [Fast tokenizers in NER](#fast-tokenizers)
- [Fast tokenizers in the QA pipeline](#fast-tokenizers-in-the-qa-pipeline)
- [Normalization and Pre-Tokenization](#normalization-and-pretokenization)
- [Byte-Pair Encoding](#byte-pair-encoding)
- [WordPiece Tokenization](#wordpiece-tokenization)
- [Unigram Tokenization](#unigram-tokenization)
- [Building a Tokenizer](#building-a-tokenizer)

## Questions/Comments
- Is that python generator just using a set?
- I've never heard of a python generator before (`yield` keyword?)
- I think it's funny the way they mask logits (give it a fat negative number)


## Training a New Tokenizer from Old
- Language model might not be available in the language you're interested in
- Corpus could be very different from the one your language model was trained on
- Will most likely want to retrain model using a tokenizer adapted to your data
- Most transformer models use a subword tokenization algorithm
    - Tokenizer needs to take a look at all the texts in the corpus
    - Training a tokenizer is a statistical process that tries to identify which subwords are the best to pick for a given corpus
    - Exact rules used to pick them differ from alg to alg
    - Not like training a model; it's deterministic

- You can train a new tokenizer with the same characteristics as an existing one using `AutoTokenizer.train_new_from_iterator()`
    - Need to transform the dataset into an iterator of lists of texts
    - Using lists of texts enables tokenization to go faster (batches vs one by one)
    - Use an iterator to avoid having everything in memory at once

Use a "python generator" to avoid loading anything into memory until necessary
```python3
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)
```
- Just creates an object you can use in a Python `for` loop
- Texts will only be loaded with you need them

Training a new tokenizer
---
- First need to load the tokenizer we want to pair model with (ex. GPT-2)
```python3
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```
- It's a good idea to avoid starting entirely from scratch
    - Won't have to specify anything about the tokenization algorithm or special tokens we want to use
    - New tokenizer will be the same as GPT-2
    - Only thing changing is vocabulary determined by the training on the corpus
- This tokenizer is originally pretty weird, can look at site if want to see actual output. In general it just treats whitespace ineffciently

Train a new tokenizer and see if it solves previous tokenizer issues:
```python3
# fixes the weird whitespace issue from the previous tokenizer
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```
- Might take a bit of time if your corpus is large
- Only works if the tokenizer you're using is a "fast" tokenizer
    - There are two types (will get into that later)
    - Fast ones are in Rust lol
    - Training a brand new tokenizer in pure Python would be incredibly slow
- Most transformer models have a fast tokenizer available
- The `AutoTokenizer` API always selects the fastest tokenizer available

Saving the tokenizer
---
Use the same `save_pretrained()` as before:
```python3
# save to device
tokenizer.save_pretrained("code-search-net-tokenizer")

# upload to hub
tokenizer.push_to_hub("code-search-net-tokenizer")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
```
- Will create a new folder called `code-search-net-tokenizer` which will contain lal the files for the tokenizer to be reloaded
- Can also push it to the hub


## Fast Tokenizers
- Slow tokenizers are written in Python inside the HuggingFace Transformers Library
- Fast tokenizers are written in Rust provided by HuggingFace Tokenizers
- Generally gotta used `batched=True` as it can be parallelized

Batch Encoding
---
- Output of a tokenizer is a `BatchEncoding` object (not a dict)
- Offset mapping = fast tokenizers always keep track of the original span of texts the final tokens come from
    - Unlocks features like mapping each word to the tokens it generated
    - Mapping each character of the original text to the token it's inside

Example:
```python3
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))

==>

<class 'transformers.tokenization_utils_base.BatchEncoding'>
```
- Get a `BatchEncoding` object in the tokenizer's output
- How to check if tokenizer is fast:
    - `tokenizer.is_fast` ==> `true`
    - `encoding.is_fast` ==> `true`

Fast tokenizers allow for token access without having to conver the IDs back to tokens
```python3
# see tokens from encoding
encoding.tokens()

==>

['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', 'and', 'I', 'work', 'at', 'Hu', '##gging', 'Face', 'in', 'Brooklyn', '.', '[SEP]']

encoding.word_ids()

==>

[None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, None]
```
- Use `word_ids()` to get the index of the word each token comes from
    - `[CLS]` and `[SEP` are mapped to `None`
    - Useful to determine if a token is at the start or if two tokens are in the same word
        - Could rely on the `##` prefix
        - `##` only works for BERT-liek tokenizers
    - This method works for any fast tokenizer
- `sentence_ids()` maps tokens to the sentence it came from (`token_type_ids` returned by tokenizer gives same information)
- Can map any word or token to characters in the original text and vice versa
    - Use the methods `word_to_chars()` or `token_to_chars()` and `char_to_word()` or `char_to_token()`
    - Ex. `word_ids()` told us that `##yl` is a word at index 3 but which word in the sentence is it?
    - Use `start, end = encoding.word_to_chars(3)` and then `example[start:end]` to get the word
- All because fast tokenizer keeps track of the span of text each token comes from in a list of offsets


Inside the token-classification pipeline
---
- Remember `pipeline()` function groups together three stages necessary to get the predictions from a raw text
- First steps in the `token-classification` pipeline are the same as in any other pipeline

```python3
from transformers import pipeline

token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# group together tokens that correspond to same entity
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
```
- Model used by default performs NER on sentences
- Can aggregate results together
    - `simple`: score is the mean of the scores of each token in the given entity
    - `first`: score of each entity is the score of the first token of that entity
    - `max`: score of each entity is the max score of the tokens in that entity
    - `average`: score of each entity is the average of the scores of the words composing that entity

First need to tokenize input and pass it rhough the model
```python3
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
```
- Get a set of logits for each token in the input sequence
- Use a softmax to convert logits to probabilities and take the argmax to get predictions
```python3
import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)

==>

[0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 6, 6, 6, 0, 8, 0, 0]
```
- The `model.config.id2label` attribute contains the mapping of indexes to labels that we can use to make sense of predictions
- In this NER, there's two types of labels, `B-type` amd `I-type`
    - `B-type`: beginning of entity
    - `I-type`: inside of entity

- This is similar to what we had before except pipeline gives info on `start` and `end` of each entity in the original sentence
```python3
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
inputs_with_offsets["offset_mapping"]
```
- `(0, 0)` is reserved for special tokens
- Info on start and end keys for each entity isn't strictly necessary
- Want to group entities together will save from messy code
```python3
import numpy as np

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]

        # Grab all the tokens labeled with I-label
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # The score is the mean of all the scores of the tokens in that grouped entity
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)
```


## Fast Tokenizers in the QA Pipeline
- How to leverage offsets to grab the answer to the question at hand from the context
- This pipeline can deal with very long contexts and will return the answer to the question even if it's at the end

- Start by tokenizing input and then sending it through the model
```python3
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)
```
- Model is trained to predict the index of the token starting the answer and the index of the token where the answer ends
- Models don't return one tensor of logits but two
    - One for the logits corresponding to the start token of the answer
    - One for the logits corresponding to the end token of the answer
- Again apply softmax function to convert logits to probabilities
    - Need to make sure to mask unwanted tokens and tokens of the question
    - Just replace logits you want to mask with a large negative number (lol)
```python3
import torch

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
mask = torch.tensor(mask)[None]

start_logits[mask] = -10000
end_logits[mask] = -10000

# apply softmax to logits
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
```

- Here could take the argmax of the start and end probabilities
    - Might end up with a `start_index` > `end_index`
    - Thus need to compute the probabilities of each possible `start_index` and `end_index` where `start_index <= end_index`
    - Then take tuple `(start_index, end_index)` with the highest probability
    - Assuminng answer starts with `start_index` and ends at `end_index` are independent, the P(answer starts at `start_index` and ends at `end_index`) = `start_probs[start_index] * end_probs[end_index]`
```python3
scores = start_probabilities[:, None] * end_probabilities[None, :]
scores = torch.triu(scores)

max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])
```
- `torch.triu()` returns the upper triangular part of the 2D tensor passed as an argument
    - Will do masking for you
- Use floor divison and modulus to get `start_index` and `end_index`
- Indices are still in terms of tokens so you need to convert to chracter indices in the context
```python3
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_char, _ = offsets[start_index]
_, end_char = offsets[end_index]
answer = context[start_char:end_char]

result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}
print(result)
```

Handling long contexts
---
- If we try to tokenize the question and long context we'll get a number of tokens higher than the max length used in the QA pipeline (384 max)
- Need to trunace inputs at the max length 
    - Don't want to truncate the question, only the context
    - Use the `only_second` truncation strategy
    - The problem with this though is that the answer might not be in the truncated context

- The QA pipeline allows to split the context into smaller chunks specifying the max length
    - Need to make sure context isn't split at the wrong place to make it possible to find the answer
    - There might be some overlap between chunks though
- Add the `return_overflowing_tokens=True` to the tokenizer argument
```python3
sentence = "This sentence is not too long but we are going to split it anyway."
inputs = tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
```
- `stride` = The overlap you want in terms of tokens
- This will make it so that each entry in the `inputs["input_ids"]` has at most 6 tokens (need padding otherwise)

Looking at `input.keys()`:
```python3
dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping'])

# tokenizing multiple sentences
sentences = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentence is shorter but will still get split.",
]
inputs = tokenizer(
    sentences, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

print(inputs["overflow_to_sample_mapping"])

==>

[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
```
- `overflow_to_sample_mapping` = a map that tells us which sentence each of the results corresponds to
- More useful when we tokenize several sentences together

```python3
inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
```
- By default, the QA pipeline has a max length of 384 and a stride of 128
- Add padding to be able to build tensors
- Can then do the steps as before like masking tokens not part of the context before taking the softmax and then using the softmax to convert logits to probabilities

- Attribute a score to all possible spans of answer, then take the span with the best score
```python3
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)

==>

[(0, 18, 0.33867), (173, 184, 0.97149)]
```
- Two candidates correspond to the best answers the model was able to find in each chunk
- Model is more confident in the second answer (as seen by the prob in the far right of the tuple)
- Can then get text that the indices correspond to like before


## Normalization and Pre-Tokenization
- Before splitting a text into subtokens (according to its model) the tokenizer:
    - First normalizes the text
    - Second performs pre-tokenizaiton

Normalization
---
- Involves general cleanup = remove whitespace, lowercasing, and/or removing accents
    - Kinda similar to Unicode normalization
- `Tokenizer` has an attribute called `backend_tokenizer` that provides access to the underlying tokenizer from the HuggingFace library
```python3
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))

==>

<class 'tokenizers.Tokenizer'>
```
- `normalizer` part of `tokenizer` has a `normalize_str()` method
    - `tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?")` ==> `'hello how are u?'`

Pre-tokenization
---
- A tokenizer cannot be trained on raw text alone
- First need to split texts into small entities like words (or BPE!)
    - Word-based tokenizers use words as the boundaries of the subtokens the tokenizer can learn during training
    - Can use the `pre_tokenize_str()` method of the `pre_tokenizer` to see pre-tokenization steps
    - Tokenizer keeps track of offsets here so it can give the offset mapping used in the previous section
- GPT-2 tokenizer replaces spaces and new lines with special symbols as seen before 
    - Will not ignore double space like BERT
    - T5 replaces white space with `_` but only splits on whitespace, not punctuation

SentencePiece
---
- Tokenization algorithm for the preprocessing of text
- Considers text as a sequence of unicode characters and replaces spaces with `_`
- Doesn't even require pre-tokenization step
- SentnecePiece is reversible tokenization
    - There is no special treatment of spaces so decoding the tokenis is done sumply but concating them and replacing the `_` with ` `. 
    - Results in the normlaized text


## Byte-Pair Encoding
- Initially developed as an algorithm to compress texts but then used by OpenAI for tokenization when pretraining GPT modles

Training Algorithm
---
- Starts by computing the unique set of words used in the corpus (after normalization and pre-tokenization)
- Builds vocabulary by taking all the symbols used to write those words
    - Base vocabulary will contain all ASCII characters and probably some Unicode characters
    - If tokenizing token is not in training corpus, character will be converted to the `[UNK]` token
    - Byte-level BPE = look at words being written with Unicode characters but with bytes
        - Base vocabulary has a small size (256) but every character will still be included and not end up being converted to `[UNK]`
- After getting the base vocabulary, add new tokens until the desired vocabulary size is reached by learning merges
    - There are rules to merge two elements of th existing vocabulary together into a new one

**The key**:
- At any step during tokenizer training, the BPE alrogithm will search for the most frequent pair of existing tokens
- The most frequent pair is the one that will be merged and we rinse and repeat for the next step

Example:
```python3
# assume text had words with the following frequencies
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

# first split each word into characters
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)

# then look at pairs (ug is most frequent so merge those)
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)

# next merge is ('h' and "ug")
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]
Corpus: ("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)

# then continue until desired vocabulary size is reached
```

Tokenization Algorithm and Implementing BPE
---
- New inputs are tokenized by applying the following steps:
    1. Normalization
    2. Pre-tokenization
    3. Splitting words into individual characters
    4. Applying merge rules learned in order on those splits

Assuming the same word set as before,
```python3
("u", "g") -> "ug"
("u", "n") -> "un"
("h", "ug") -> "hug"
```
- `bug` = `b` + `ug`
- `mug` = `[UNK]` + `ug`
- `thug` = `[UNK]` + `hug`

Now implementing BPE:
```python3
from collections import defaultdict

# compute word frequencies
word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

# compute base vocabulary formed by all the characters used in the corpus
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

# add special end of text token at the beginnig
vocab = ["<|endoftext|>"] + alphabet.copy()

# split each word into characters to be able to start training
splits = {word: [c for c in word] for word in word_freqs.keys()}

# compute the frequency of each pair
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

# find most frequent pair
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

# merge the most frequent pair
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

# now train for a vocab size of 50
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
```
- Vocab is composed of the special token, the initial alphabet, and all of the results of the mergesw
- Using `train_new_from_iterator()` on the same corpus won't result in the exact same vocabulary
    - When there is a choice of the most frequent pair, we select the first one encountered
    - The HuggingFace Tokenizers library sepects the first one based on its inner IDs

Now apply new tokenizer to text
```python3
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

tokenize("This is not a token.")

==>

['This', 'Ġis', 'Ġ', 'n', 'o', 't', 'Ġa', 'Ġtoken', '.']
```
- This implementation will throw an error if there's an unknown character since we didn't do anything to handle them
- Did not include all possible bytes in the initital vocabulary
- Make sure to handle it when doing it fr!


## WordPiece Tokenization
- Alrogithm developed by Google to pretrain BERT
- Similar to BPE in terms of training, but the actual tokenization is done differently

Training Algorithm
---
- Starts with a small vocabulary including the special tokens used by the model and initial alphabet
- Identifies subwords by adding a prefix (like `##` for BERT)
    - Each word is initially split by adding that prefix to all the characters inside the word
    - Ex. `word` = `w ##o ##r ##d`
- Thus, initial alphabet is all the characters present at the beginning of a word and the caracters inside a word with the WordPiece prefix
- WordPiece also learns merge rules like BPE
    - The way the pair to be merged is selected using a score
    - `score = (freq_of_pair)/(freq_of_first_element×freq_of_second_element)`
    - Algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary
    - Ex. even if `un` and `##able` occurs frequently, they won't necessairly merge as `un` and `##able` will likely each appear in a lot of other wirds with high frequency

Tokenization Algorithm
---
- Only saves the final vocabulary, not the merge rules learned
- Finds the longest subword in the vocabulary then splits it
    - With BPE, would have applied the merges learned in order and tokenized it so the encoding is different
- When tokenization gets to a stage where it's not possible to find a subword in the vocabulary, the whole word is tokenized as unknown (`[UNK]`)
    - Different from BPE which would only classify the individual characters not in the vocabulary as unknown

Implementing WordPiece:
```python3
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# need to pre-tokenize corpus into words
from transformers import AutoTokenizer

# use bert-base-case tokenizer for pre-tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# compute frequencies of each word in the corpus
from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

# create alphabet
alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()

# want to add special tokens used by the model a the beginning of the vocab
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()


# need to split each word with all the letters not prefixed by ##
splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in word_freqs.keys()
}



# compute score for each pair
def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores


# look at a part of dict after initial splits
pair_scores = compute_pair_scores(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f"{key}: {pair_scores[key]}")
    if i >= 5:
        break

# find pair with the best score
best_pair = ""
max_score = None
for pair, score in pair_scores.items():
    if max_score is None or max_score < score:
        best_pair = pair
        max_score = score


# need to apply the merge in splits dict
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


# need to loop until we have learned all the merges we want 
vocab_size = 70
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)


# Then to tokenize a new text, pre-tokenize it, split it, then apply tokenization algorithm onto each word
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens

# tokenize a text
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])
```


## Unigram Tokenization
- Algorithm used in SentencePiece ==> tokenization algorithm used by AlBERT, T5, mBART, Big Bird, etc.
- Unlike BPE and WordPiece, it starts from a big vocabulary and removes tokens until it reaches the desired vocab size
- Build base vocabulary by taking most common substrings in the pre-tokenized words or apply BPE on the initial corpus with a large vocab size

Training Algorithm
---
- Algorithm computes a loss over the corpus given the current vocabulary
- Then for each symbol in the vocab, the alrogithm computes how much the overall loss would increase if the symbol was removed and looks for the symbols that would increast it the least
    - Those symbols have a lower effect on the overall loss of the corpus so they are in a sense "less needed" and are the best candidates for removal
    - Is a very costly operation so use the `p` hyperparameter to control the percent of symbols to remove associated with the lowest loss scores
- Never remove base characters
- Main part in algorithm is to compute a loss over the corpus and see how it changes when you remove some other tokens in the vocabulary

Tokenization Algorithm
---
- Considers each token to be independent of the tokens before it
    - P(token X | tokens before it) = P(token X)
    - If we used a unigram language model to generate text, would need to always predict the most common token

Ex. Take the following corpus `("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)` which has the following substrings `["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]`
- The probability of a given token is its frequency in the original corpus / sum of all frequencies of all tokens in the vocabulary

If all possible subwords have frequencies `("h", 15) ("u", 36) ("g", 20) ("hu", 15) ("ug", 20) ("p", 17) ("pu", 17) ("n", 16) ("un", 16) ("b", 4) ("bu", 4) ("s", 5) ("hug", 15) ("gs", 5) ("ugs", 5)`
    - Sum of all frequencies = 210. This P("ug") = 20/210

- To tokenize a given word, look at all possible segmentations into tokens and compute the probability of each according to the Unigram model
    - Due to independence, the probability is just the product of the probability of each token
- In general, tokenizations with the least tokens possible will have the highest probability
    - Corresponds to what we want intuitively: Split a word into the least number of tokens possible

Ex. when tokenizing `pug`
```python3
["p", "u", "g"] : 0.000389
["p", "ug"] : 0.0022676
["pu", "g"] : 0.0022676
```
- `pug` would be tokenized as `["p", "ug"]` or `["pu", "g"]` depending on which is encountered first

- In general it's going to be harder to find all the possible segmentations and compute their probabilities
    - Use the *Viterbi algorithm* to build a graph to detect the possible segmentations of a given word by saying there is a branch from a character `a` to a character `b` if the subword from `a` to `b` is in the vocab
    - Then attribute to that branch the probability of the subword
    - Algorithm determines for each position in the word, the segmentation wit hthe best score that ends at that position
    - Best score can be found by looping through all the subwords ending at the current position and then using the best tokenization score from the position this subword begins at
    - Then just have to unroll the path taken to arrive at the end

Ex. Tokenizing `unhug`
```python3
Character 0 (u): "u" (score 0.171429)
Character 1 (n): "un" (score 0.076191)
Character 2 (h): "un" "h" (score 0.005442)
Character 3 (u): "un" "hu" (score 0.005442)
Character 4 (g): "un" "hug" (score 0.005442)
```
- Thus `unhug` would be tokenized as `["un", "hug"]`


Back to Training
---
- Loss is computed by tokenizing every word in the corpus, using the current vocabulary and the Unigram model determined by the frequenceis of each token in the corpus
- Each word has a score and the loss is the negative log likelihood of the scores (`-log(P(word))`)

Ex. going back to the corpus
```python3
# this guy
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

# The tokenization of each word with their respective score is:
"hug": ["hug"] (score 0.071428)
"pug": ["pu", "g"] (score 0.007710)
"pun": ["pu", "n"] (score 0.006168)
"bun": ["bu", "n"] (score 0.001451)
"hugs": ["hug", "s"] (score 0.001701)

# so the loss is:
10 * (-log(0.071428)) + 5 * (-log(0.007710)) + 12 * (-log(0.006168)) + 4 * (-log(0.001451)) + 5 * (-log(0.001701)) = 169.8

# then need to compute how removing each token affects the loss (tedious so will just do it for two tokens and save the whole process for later
# removing "hug" will make the loss worse as:
"hug": ["hu", "g"] (score 0.006802)
"hugs": ["hu", "gs"] (score 0.001701)

These changes will cause the loss to rise by: 
- 10 * (-log(0.071428)) + 10 * (-log(0.006802)) = 23.5
```
- Therefore the token `pu` will be removed from the vocabm not `hug`

Now there's a whole section here about implementing Unigram. It's a lot of the same as before with WordPiece and BPE so I'm not going to type it all out. Its here if I want to fully implement it though.


## Building a Tokenizer
As we've seen, tokenization requires several steps:
1. Normalization
    - Cleanup of text like whitespace removal, removing accents, unicode normalization, etc.
2. Pre-tokenization
    - Splitting the input into words
    - List of words
3. Running input through the model
    - Using the pre-tokenized words to produce a sequence of tokens
4. Post-processing
    - Adding the special tokens of the tokenizer
    - Generating the attention mask and token type IDs

Transformers library is built around a central `Tokenizer` class regrouped in submodules
- `normalizers`: contains all the possible types of `Normalizer`
- `pre_tokenizers`: contains all the possible types of `PreTokenizer`
- `models`: contains the various types of `Model` you can use (`BPE`, `WordPiece`, `Unigram`)
- `trainers`: contains lal the dufferent types of `Trainer` you can use to train your model on a corpus
- `post_processors`: contains the various types of `PostProcessor` you can use
- `decoders`: contains the various types of `Decoder` you can use to decode the outputs of the tokenization

Building a WordPiece tokenizer from scratch
---
- Use the WikiText dataset and BERT model
```python3
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
```
- Have to specify the unknown token
- Can also include the `vocab` of the model and `max_input_chars_per_word`

```python3
# implement normalizer
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

# or if you want to do it from scratch:
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
```
- There's already a BERT normalizer with the classic options set for BERT
- `lowercase` and `strip_accents` are default, `clean_text` removes all control characters and replaces repeating spaces with a single one
    - There's also `handle_chinese_chars` which places spaces around Chinese characters
- `NFD` = Unicode normalizer

```python3
# Then use a pre tokenizer 
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# or build it from scrach
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# split on ONLY whitespace
pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# again can manually build it with sequence
pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
```
- Again there is a prebuilt `BertPreTokenizer`
- `Whitespace` splits on whitespace and all characters that aren't alphanumeric or underscore
    - Technicially it means split on punctuation too

```python3
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

# can use text files to train tokenizer
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)

# test tokenizer by calling the encode() method
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
```
- Then run inputs through the model
- Need to pass the trainer with all the special tokens you intend to use otherwise they won't be added in the vocabulary
- In addition to `vocab_size` and `special_tokens`, can also set the `min_frequency`=number of times a token must appear to be in the vocab or `continuing_subword_prefix`=want to use something other than `##`
- `Encoding` conatins all necessary outputs in the tokenizer in its various attributes
    - `ids`, `type_ids`, `tokens`, `offsets`, `attention_mask`, `special_tokens_mask`, `overflowing`

```python3
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

# again test using encoding to see the tokens produced
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)

==>

['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '...', '[SEP]', 'on', 'a', 'pair', 'of', 'sentences', '.', '[SEP]']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```
- Last step is the post-processing
- Use a `TemplateProcessor` to add special tokens to beginning and end of sentneces
    - Have to specify how to treat a single sentence and a pair of sentences
    - Write the speical tokens we want to use (first token = $A, second sentence = $B)

```python3
tokenizer.decoder = decoders.WordPiece(prefix="##")

tokenizer.decode(encoding.ids)

==>

"let's test this tokenizer... on a pair of sentences."
```
- Last step is a decoder to test

To use tokenizer in HuggingFace Transformers, need to wrap it in a `PreTrainedTokenizerFast`
```python3
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# if using a specific tokenizer class, need to specify the special tokens that are different from the default ones
from transformers import BertTokenizerFast

wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
```
- Can either pass the tokenizer as a `tokenizer_object` or the tokenizer file you can save (use `tokenizer.save(filename)`) as a `tokenizer_file`

The tutorial then does this for both BPE and Unigram tokenizers. It's a lot of the same stuff so I'm not going to take notes on it but its here if I need.
