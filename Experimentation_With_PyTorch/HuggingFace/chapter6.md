## Hugging Face Chapter 5 Notes

## Module 5 - The Tokenizers Library

- [Training a New Tokenizer from Old](#training-a-new-tokenizer-from-old)
- [Fast tokenizers in NER](#fast-tokenizers)
- [Fast tokenizers in the QA pipeline](#fast-tokenizers-in-the-qa-pipeline)
- [Normalization and Pre-Tokenization](#normalization-and-pretokenization)

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
