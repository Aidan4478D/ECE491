## Hugging Face Chapter 2 Notes

## Module 2 - Using Hugging Face Transformers

- [Behind the Pipeline](#behind-the-pipeline)
- [Models](#models)
- [Tokenizers](#tokenizers)
- [Handling Multiple Sequences](#handling-multiple-sequences)
- [Putting it All Together](#putting-it-all-together)

## Questions
- Can we clarify Sequence length: The length of the numerical representation of the sequence (16 in our example).
- Logits as output of transformer to be put through a softmax

## Behind the Pipeline
---
- Pipeline grups together three steps: preprocessing, passing the inputs through the model, and postprocessing

- Preprocessing
    - Transformers can't process raw text directly
    - Use a tokenizer to split the input into words/subwords (tokens) and map each token to an integer
    - Preprocessing needs to be done exactly in the same way as when the model was pretrained
        - need to download info from the model hub
        - AutoTokenizer and from_pretrained()
        - will automatically fetch the data associatd with the model's tokenizer and cache it
    - Once we have the tokenizer, we can directly pass our sentences to it and we’ll get back a dictionary that’s ready to feed to our model

    - Transformer models only accept `tensors` as input
        - Kinda similar to NumPy arrays
        - Can specify type of tensors (PyTorch, TensorFlow, NumPy) `inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")`
    
    - PyTorch tensors example:

```python
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```
- `input_ids` contains a row of ints for each sentence that are unique identifiers of the tokens in each sentence.
- `attention_mask` is something explained later on in the chapter

- Outputs *hidden states* aka *features* = contextual understanding of that input by the transformer model
- Hidden states are usually inputs to another part of the model (known as the head)
- Different tasks could be performed by the same architecture but each of these tasks will have a different head associated with it.

**Outputs by Transformer Module**
---
- Usually very large, consisting of three dimensions:
    - Batch size: Number of sequences processed at a time
    - Sequence length: Length of the numerical representation of the sequence
    - Hidden size: The vector dimension of each model input

- Considered to be "high dimnesional* because of the hidden states (768 common for smaller models, 3072 for larger models)
- Outputs of the Hugging Face transformers behave like `namedtuples` or dictionaries

**Model heads**
---
- Model head takes the high-dimensional vector of hidden states as input and projects them onto a different dimension
- Usually composed of one or a few linear layers
- Output of the Transformer model is sent directly to the model head to be processed

- Converts the transformer predictions to a task-specific output

- Embeddings layer converts each input ID in the tokenized input into a vector that represents the associated token
- Subsequent layers manipulate those vectors using the attention mechanism to produce the final representation of the sentences

**Postprocessing**
---
- Values from model as output don't necessarily make sense by themselves

```python
print(outputs.logits)

==>

tensor([[-1.5607,  1.6123], [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)

```
- In this case for their example, the model predicted [-1.5607, 1.6123] for the first sentence and [ 4.1692, -3.3464] for the second one.
- Not probabilities but `logits` = raw, unnormalized scores outputted by the last layer of the mode
- Need to go through a softmax layer to convert to probabilities
- All hugging face transformer models output logits

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

==>

tensor([[4.0195e-02, 9.5980e-01], [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

- Now can see that the model predicted [0.0402, 0.9598] for the first sentence and [0.9995, 0.0005] for the second one.
- can use the `model.config.id2label` to get the labels corresponding to each position

**Overall**
---
- Three steps of pipelines
    - Preprocessing with tokenizers
    - Passing inputs through the model
    - Postprocessing with a softmax

## Models
---
- `AutoModel` lets you instantiate any model from a checkpoint
- The `AutoModel` class and all of its relatives are actually simple wrappers over the wide variety of models available in the library
- Can also use the class that defines its architecture directly

Ex. initalizing a BERT model
```python
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
```
- Creating a model from the default configuration initializes it with random values
- Model can be used in this state but its output will be gibberish
    - Needs to be trained first
    - Can also just load it from a checkpoint using `from_pretrained`

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```
- `bert-base-cased` is a model checkpoint trained by the authors of BERT
- Is initialized with the weights at that checkpoint
- Weights are downloaded and cached
- Identifier used to load the model can be the identifier of any model on the model hub as long as it's compatabile with the BERT architecture


**Saving Methods**
---
- Saving a model is as easy as loading one

```python
model.save_pretrained("directory_on_my_computer")
```
- Saves `config.json` and`pytorch_model.bin` onto disk
- `config.json` = attributes necessary to build the model architecture
- `pytorch_model.bin` = state dictionary; contains all model's weights

**Using model for inference**
---
- Transformers can only process numbers (duh) that tokenizer generates
- Tokenizers can take care of casting the inputs to the appropriate framework's tensors

ex.
```python
sequences = ["Hello!", "Cool.", "Nice!"]

==> 

#turn into list of lists
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

==>

import torch

# convert lists to tensor
model_inputs = torch.tensor(encoded_sequences)

==>

# make use of model with inputs
output = model(model_inputs)
```

- Tokenizer converts these to vocabulary indices which are typically input IDs
- Each sequence turns into a list of lists of ints
- Tensor only accepts rectangular shapes
- Model can accept a lot of args but only the input IDs are necessary



## Tokenizers
- Translate text into data that can be processed by the model
- Need to convert text input to numerical data
- Goal is to find the most meaningful representation that makes the most sense to the model (and if possible the smallest)

Word Based
---
- Split text into words and find a numerical representation for each of them
- Can split text on whitespace using `split()`
- Can end up with pretty large vocabularies = total number of independent tokens in the corpus
- Each word gets a unique ID from 0 - len(vocabulary)
- Also need a token to represent words not in the vocabulary
    - Goal is to do it in such a way that the tokenizer tokenizes as few words as possible into unknown tokens

Character Based
---
- Split text into characters rather than words
    - Vocabulary is much smaller
    - There are much fewer out-of-vocabulary tokens (unknown)
    - Questions arise concerning spaces and punctuation
- Representation is now based on chractesr rather than words
    - Each character doesn't mean a lot on its own
    - Depends from language to language
- Will end up with a large amount of tokens to be processed by model

Subword Based
---
- Rely on the principle that frequently used words should not be split into smaller tokens
- Rare words should be decomposed into meaningful subwords
- Both tokens can have semantic meaning wile being space-efficient
- Allows for relatively good coverage with small vocabularies and close to no unknown vocabularies

Other methods
---
- A couple more techniques
    - Byte-pair encoding (BPE) used in GPT2
    - WordPiece used in BERT
    - SentencePiece or Unigram used in multilingual models

Loading & Saving
---
- Load and save just like models
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```
- Loading BERT tokenizer as well as its vocabulary

```python
tokenizer("Using a Transformer network is simple")

==>

{
    `input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

- **Encoding** = translating text to numbers
    - Tokenization ==> convert to input IDs
- Need to use same tokenizer as what model was trained on
- Then convert tokens to numbers so we can build a tensor out of them
    - Tokenizer has a vocabulary where it maps the tokens to ints
    - One converted to the appropriate framework tensor, it can be used as inputs to a model

- **Decoding** = going from vocabulary indices to strings
    - Not only converts indices back to tokens but also groups together tokens that were part of the same words
    - Useful for models that predict new text


## Handling Multiple Sequences
- Saw how we can use models for a single sequence with small length
- How can we handle multiple sequences of different lengths though?

Batches
---
- HuggingFace transformer models expect multiple sentences by default
- Tokenizer doesn't just conver the list of input IDs into a tensor, it also adds a dimension on top of it (list of lists)

example:
```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

==>

Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
Logits: [[-2.7276,  2.8789]]
```

- **Batching** = Act of sending multiple sentences through the model all at once
```python
batched_ids = [ids, ids]
```
- This will batch two identical sequences
- Allows the model to work when you feed it multiple sentences
- Using multiple sequences is as simple as building a batch with a single sequence
- Batches might have different lengths but tensors need to be of rectangular shape

Padding
---
Following cannot be converted to a tensor
```python
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```
- Need to use padding to make sure tensors have a rectangular shape
- `Padding` = makes sure sentences have the same length by adding a special `padding token`

```python
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```
- Padding ID token is found in `tokenizer.pad_token_id`
- Sentences with and without padding might have different logits
    - Attnetion layers *contextualize* each token
    - Takes into acconunt he padding tokens since they attend to all of the tokens in a given sequence
    - Need to tell the attention layers to ignore the padding tokens, done with an attention mask

Attention Masks
---
- Tensors with the exact shape as the input IDs tensor filled with 0s and 1s
- 0 = token should not be attended to (ignored by attention layer)
- 1 = token should be attended to
- Usually 0 for padding tokens

Longer Sequences
---
- There is a limit to the lengths fo the sequences we can pass the models
- Most models handle sequences of up to 512 or 1024 and will crash when asked to process longer sequences
    - Can use a model with a longer supported sequences
    - Can truncate your sequences
- Models like `Longformer` or `LED` are built if you need long sequences
- Otherwise can truncate with `sequence = sequence[:max_sequence_length]`

## Putting it All Together
- When you call `tokenizer` directly on the sentence, you get back inputs that are ready to pass through the model
    - input ids, padding, truncation, and attention masks
- `tokenizer` can tokenize a single sentence as well as multiple sequences at a time with no change in the API
- Can pad according to several objects and truncate
```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)


sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

- Can also handle conversion to specific framework of tensors that can be directly sent to the model
```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

- Some models add special tokens at the beginning of the sequence `[CLS]` or at the end `[SEP]`
- Model was pretrained with those so to get the same results for inference, we need to add them as well
- `tokenizer` knows which ones are expected and will deal with this for you


