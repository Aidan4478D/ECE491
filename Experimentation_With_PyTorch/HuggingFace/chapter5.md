## Hugging Face Chapter 5 Notes

## Module 5 - The Datasets Library

- [Dataset not on Hub](#dataset-not-on-hub)
- [Slice and Dice Data](#slice-and-dice-data)
- [Big Data](#big-data)

## Questions/Comments
- I think the "Kinny rule of thumb" is interesting - 5-10x as much RAM as the size of your dataset
- Also datasets as memory mapped files is cool

## Dataset not on Hub
- Will often work with data stored on laptop or on a remote server
- HuggingFace `Datasets` provides loading scripts to handle the loading of local and remote datasets
    - CSV: `load_dataset("csv", data_files="my_file.csv")`
    - Text: `load_dataset("text", data_files="my_file.txt")`
    - JSON: `load_dataset("json", data_files="my_file.jsonl")`
    - Pickle: `load_dataset("pandas", data_files="my_dataframe.pkl")`

- Load a local dataset
```python3 
from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")

```
- By default loading local files creates a `DatasetDict` object with a `train` split
```python3
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset
```
- Use `Dataset.map()` function to generate both train and test split (if we downloaded it like that)
- Supports automatic decompression of the input files so we could have skipped the use of `gzip` by pointing the `data_files` argument directly to the compressed file

- Loading a remote dataset:
```python3
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```
- Returns the same `DatasetDict` object obtained above but sames the step of manually downloading and decompressing the files

## Slice and Dice Data
- Hugging face `Datasets` has several functions to manipulate contents of the `Dataset` and `DatasetDict` objects

First have to import data:
```python3
from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# shuffle data and look at first few examples
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
drug_sample[:3]
```
- `Dataset.select()` expects an iterable of indices, so we’ve passed `range(1000)` to grab the first 1,000 examples from the shuffled dataset.

```python3
# verify number of IDs matches the number of rows in each split
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

# rename the Unnamed: 0 column
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
```

- Create lambda functions in python like: `lambda <arguments> : <expression>`
    - To apply the function to an input, need to wrap it and the input in parentheses
    - Ex. `(lambda x: x * x)(3)` ==> `9`
    - Can also do something like `(lambda base, height: 0.5 * base * height)(4, 8)` ==> `16.0`
    - Handy when you want to define small, single-use functions

- Lambda functions are useful for us if we want to do simple map and filter operations
```python3
# using Dataset.filter() to remove rows where condition is None
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
drug_dataset["train"]["condition"][:3]
```

Creating new columns
---
- In this case, using the `Dataset.map` function
```python3
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
drug_dataset["train"][0]

==>

{
    'patient_id': 206461,
    'drugName': 'Valsartan',
    'condition': 'left ventricular dysfunction',
    'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
    'rating': 9.0,
    'date': 'May 20, 2012',
    'usefulCount': 27,
    'review_length': 17
}

===
drug_dataset["train"].sort("review_length")[:3]
```
- `review_length` column added to training set
- Can use `Dataset.sort()` to look at extreme values

Dataset.map()
---
- Takes a `batched` argument that, if set to True, causes it to send a batch of examples to the map function at once
- When you specify `batched=True` the function recieves a dict with the fields of the dataset but each value is now a list of values and not a single value

- Using the map function kinda seems like vectorization vs for loops in MATLAB
    - Faster than executing the same code in a `for` loop
```python3
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)
```
- Using a fast tokenizer with the `batched=True` option is 30 times faster than its slow counterpart with no batching
- Tokenization code is executed in Rust making it easy to parallelize code

- Use `num_proc` argument and specify number of processes to use to enable multi-processing
```python3
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)

def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)

tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
```
- Using `num_proc` to speed up your processing is usually a great idea, as long as the function you are using is not already doing some kind of multiprocessing of its own.

Datasets to Dataframes and back
---
- Datasets provides a `Dataset.set_format()` function
- Function only chanes the output format of the dataset so you can switch to another format without affecting the underlying data format

- `Dataset.set_format()` changes the return format for the dataset's `__getitem__()` method

```python3
drug_dataset.set_format("pandas")
train_df = drug_dataset["train"][:]

frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
frequencies.head()
```

- Once done with Pandas analysis, can create a new `Dataset` object and convert back to `Dataset`
```python3
from datasets import Dataset

freq_dataset = Dataset.from_pandas(frequencies)
freq_dataset

==>

Dataset({
    features: ['condition', 'frequency'],
    num_rows: 819
})
```

Saving a Dataset
---
- Three main functions to save dataset in different formats
    - Arrow: `Dataset.save_to_disk()`
    - CSV: `Dataset.to_csv()`
    - JSON: `Dataset.to_json()`
- Arrow format as a fancy table of columns and rows that is optimized for building high-performance applications that process and transport large datasets.
- Can then use `load_from_disk()` function to reload file back into program


## Big Data
- Treats datasets as memory-mapped files
- Combats hard drive limits by streaming the entries in a corpus

- The Pile = English text corpus for training large-scale language models
    - Training corpus is available in 14 GB chunks

- Can measure memory usage in python with `!pip install psutil`
    - Provides a `Process` class that allows to check memory usage of the current process

- Datasets treats each dataset as a memory-mapped file
    - Provdes a mapping between RAM and filesystem storage that llows the library to acces and operatoe on elements of the dataset without needing to fully load it into memory
    - Can be shared across multiple process (like a mapped anonymous region?)
    - Enables `Dataset.map()` to be parallelized without needing to move or copy the dataset

Streaming datasets
---
- Just need to pass the `streaming=True` arg to the `load_dataset()` function
```python3
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
```
- The object returned is an `IterableDataset`
    - To access these elements you need to iterate over it
```python3
next(iter(pubmed_dataset_streamed))

==>

{'meta': {'pmid': 11409574, 'language': 'eng'},
 'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection.\nTo determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age ...'}
```
- Elements can be processed on the fly using `IterableDataset.map()`
    - Useful for training of you need to tokenize the inputs
    - Process is same as in chapter 3, just outputs are returned one by one
- Can shuffle a streamed dataset using `IterableDataset.shuffle()` 
    - This only shuffles elements in a pre defined `buffer_size`

```python3
#select a random sample in the 10,000 example buffer
shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
next(iter(shuffled_dataset))
```

- Once an example is accessed, its spot in the buffer is filled with the next example in the corpus
- Can also select elements from a stread dataset using `IterableDataset.take()` and `IterableDataset.skip()`
```python3
# take first five examples in the PubMed Abstracts dataset
dataset_head = pubmed_dataset_streamed.take(5)
list(dataset_head)

# Skip the first 1,000 examples and include the rest in the training set
train_dataset = shuffled_dataset.skip(1000)
# Take the first 1,000 examples for the validation set
validation_dataset = shuffled_dataset.take(1000)
```
- Create training and validation splits from a shuffled dataset

- `interleave_datasets()` function converts a list of `IterableDataset` objects into a single `IterableDataset`
    - Elements of the new dataset are obtained by alternating among the source examples
    - Useful when trying to combine large datasets
```python3 
from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
list(islice(combined_dataset, 2))
```
- Use the `islice()` funcition to select the first two examples from the combined dataset


