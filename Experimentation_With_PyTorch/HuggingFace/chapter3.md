## Hugging Face Chapter 3 Notes

## Module 3 - Fine Tuning a Pre-Trained Model

- [Processing the Data](#processing-the-data)
- [Fine tuning with the Trainer API](#fine-tuning-with-the-trainer-api)
- [A Full Training](#a-full-training)

## Questions/Comments
- The data collator with dynamic padding is cool

## Processing the Data 

How to train a sequence classifier on one batch in PyTorch:
```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```
- Want to train models on more than two sentences tho lol
- The HuggingFace Hub also contains datasets in lots of languages as well as models

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")

==>

DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```
- Downloads a `DatasetDict` object that is a dict of train, test, and validate `Dataset` objects
- Don't have to do any preprocessing with these datasets
- Can consult `raw_train_dataset.features` to see what each feature is

Preprocessing a dataset
---
- Need to convert the text to numbers the model can make sense of
- Use a tokenizer as seen in previous chapters
- Tokenizer can take in a pair of sequences and prepare it the way a bert model expects

```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs

==>

{ 
    'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```
- `token_type_ids` tells the model which part of the input is the first sequence and which is the second
- Has a special `[SEP]` token at the end of each sequence but only one `[CLS]` token
- BERT is pretrained with token type IDs
    - One of the pre-training objectives is also NSP (known)
- Generally don't need to worry about whether or not there are `token_type_ids` in tokenized inputs
    - As long as you use the same checkpoint for the tokenizer and model, everything will be fine

Can also feed lists of pairs of sentences to tokenizer to train a lot at the same time
```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```
- Returns a dictionary
- Only works if you have enough RAM to store the whole dataset during tokenization
- Use the `Dataset.map()` method to keep the data as a Dataset
    - Allows for more flexibility if we need more preprocessing done than just tokenization
    - `map()` works by applying a function on each element on the dataset
    - Takes a dictionary and returns a new dict with the keys `input_ids`, `attention_mask`, and `token_type_ids` like in the `Dataset` object
- Padding all the samples to the max length is not efficient
    - It's better to pad samples when building a batch as then you only need to pad to the max length in the batch and not the max length in the dataset
    - Allows for a lot faster preprocessing
    - Called **dynamic padding**

- **Collate function** = function that puts together samples in a batch
    - Is an argument you can pass when building a `DataLoader`
    - Default function will convert samples to PyTorch tensotrs and concats them
    - Not possible if inputs won't all be of the same size (as we have delayed the padding)

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
- Will apply the correct amount of padding to the items of the dataset we want to batch together
- `DataCollatorWithPadding` takes a tokenizer and will do anything you need

## Fine-tuning with the Trainer API
- The HuggingFace library provides a `Trainer` class to help fine-tune any of the pretrained models
- Hardest part is likely preparing the environment to run `Trainer.train()`
    - Will run very slowly on a CPU
    - Can try running on Google Colab tho

Training
---
- Need to define `TrainingArguments` to set all the hyperparameters for the `Trainer` to use 
- Only argument is a directory where the trained model will be saved as well as checkpoints along the way
```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```
- Then have to define a model (any on Hugging face)
- Then define trainer by passing all the objects constructed up to now
```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```
- Default data collator is `DataCollatorWithPadding`
- Can then just run `trainer.train()`
    - Will start the fine-tuning and report training loss every 500 steps
    - Won't tell you how model is performing as:
        1. Didn't tell `Trainer` to evaluate during training steps (set `evaluation_strategy` to `steps` or `epoch`)
        2. Didn't provide `Trainer` with `compute_metrics()` function to calculate a metric during said evaluation

Evaluation
---
- `compute_metrics()` function must take an `EvalPrediction` object (named tuple with a `predictions` and `label_ids` field)
    - Will return a dict mapping strings (names of metrics) to floats (metric values) and `metrics` (loss on the dataset passed and time to predict)

```python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```
- Now within the `Trainer` initialization, use `compute_metrics=compute_metrics`
- Need to create a `TrainingArguments` with its `evaluation_strategy` set to `"epoch"` and a new model
    - Otherwise would just be continuing the training of the model we have already trained
- The exact accuracy/F1 score you reach might be a bit different from what we found, because of the random head initialization of the model, but it should be in the same ballpark.
- The `Trainer` will work out of the box on multiple GPUs or TPUS

## A Full Training
