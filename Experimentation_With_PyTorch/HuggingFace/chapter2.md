## Hugging Face Chapter 2 Notes

### Module 2 - Using Hugging Face Transformers

- [Behind the Pipeline](#behind-the-pipeline)
- [Models](#models)

### Questions
- Can we clarify Sequence length: The length of the numerical representation of the sequence (16 in our example).

### Behind the Pipeline
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

### Models
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
