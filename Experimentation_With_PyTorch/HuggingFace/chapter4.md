## Hugging Face Chapter 4 Notes

## Module 4 - Sharing Models & Tokenizers

- [The Hugging Face Hub](#the-hugging-face-hub)
- [Using Pretrained Models](#using-pretrained-models)
- [Sharing Pretrained Models](#sharing-pretrained-models)
- [Building a Model Card](#building-a-model-card)

## Questions/Comments

## The Hugging Face Hub
- A central platform that enables anyone to use models and datasets
- Link to the hub: [The Hub](https://huggingface.co/)
    - 10,000+ publically available models
    - Models from Flair, AllenNLP, Asteroid, etc.
- Each model hosted as a Git repository
- Automatically deploys a hosted inference API for that model
- Sharing and using any public model on the hub is completely free

## Using Pretrained Models
- Model hub makes selecting the appropriate model simple

Example using the `camembert-base` model checkpoint
```python
from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")

==>

[
    {'sequence': 'Le camembert est délicieux :)', 'score': 0.49091005325317383, 'token': 7200, 'token_str': 'délicieux'}, 
    {'sequence': 'Le camembert est excellent :)', 'score': 0.1055697426199913, 'token': 2183, 'token_str': 'excellent'}, 
    {'sequence': 'Le camembert est succulent :)', 'score': 0.03453313186764717, 'token': 26202, 'token_str': 'succulent'}, 
    {'sequence': 'Le camembert est meilleur :)', 'score': 0.0330314114689827, 'token': 528, 'token_str': 'meilleur'}, 
    {'sequence': 'Le camembert est parfait :)', 'score': 0.03007650189101696, 'token': 1654, 'token_str': 'parfait'}
]
```
- Need to keep in mind if the chosen checkpoint is suitable for that task it's going to be used for
- `text-classification` pipeline is different from the `fill-mask` pipeline
- Can use the "task selector" in the Hugging Face Hub to select the appropriate checkpount

Can also instantiate the checkpoint using the model architecture directly
```python
from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
```
- Recommend using the Auto classes instead (Ex. `AutoTokenizer`, `AutoModelForMaskedLM`, etc.)
    - Are design architecture agnostic
- Make sure to check out how model was trained, on which datasets, its limits, and biases

## Sharing Pretrained Models
- Can train and share models onto the hub
- To create new model repositories, can either:
    1. Use the `push_to_hub` API
    2. Use the `huggingface_hub` Python library
    3. Use the web interface

Push To Hub API
---
- Need to generate an authentication token so the `huggingface_hub` API know who you are and what namespaces you have write access to
- Need `transformers` installed

- Enter the code: 
```python3 
from huggingface_hub import notebook_login
from transformers import TrainingArguments

notebook_login()

training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
)
```
- And log into hugging face with `huggingface-cli login` in the terminal
- Easiest way to upload to hub is to set `push_to_hub=True` when defining `TrainingArguments`
- Trainer will then upload model to the hub each time it's saved when running `Trainer.train()`
    - Can also set `hub_model_id` = `org_name/repo_name`
- Once training finished, should do a final `trainer.push_to_hub()` to upload last version of your model
    - Also generates model card with all relevant metadata
    - Reports hyperparameters and evaluation results

- Accessing the Model Hub can be done directly on models, tokenizers and config objects through their `push_to_hub()` method
    - Takes care of both the repository creation and pushing the model and tokenizer files directly to the repository
    - No manual handling is required

Something like:
```python
model.push_to_hub("dummy-model")
```
- Creates new repository `dummy-model` in your profile and populates it with your model files

```python3
tokenizer.push_to_hub("dummy-model", organization="huggingface", use_auth_token="<TOKEN>")
```
- Can also push something like the tokenizer, set organization, and auth token

Using the huggingface_hub Python library
---
- Package that offers a set of tools for the model and dataset hubs
- Provides simple methods and classes for common tasks like getting information about repos on the hub and managing them
- Will require you to have your API token saved in cache (so use the `huggingface-cli login` command again like before)

Offers several methods and classes for repository creation, deletion, and others like:
```python3
from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,

    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,

    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)
```

- Offers a powerful `Repository` class to manage a local repository
```python3
from huggingface_hub import create_repo

create_repo("dummy-model", organization="huggingface")
```
- Will create the repo `dummy-model` in your namespace
- Can speciy which organization and other parameters like `private`, `token` (override the token stored in cache) or `repo_type` (want to create a dataset or space instaed of a model)

- The web interface seems pretty simple so I'm not gonna take notes on it
- Seems exactly like a git repo

Uploading model files
---
- Based on git for regular files and git-lfs (large file storage) for larger files

Upload file approach: 
```python3
from huggingface_hub import upload_file

upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)
```
- Does not require git and git-lfs to be installed on your system
- Pushes files directly to hugging face hub directly through HTTP POST requests
    - Can't handle files > 5GB
- Will upload a `config.json` file to the root of rhe repository

Repository class:
```python3
from huggingface_hub import Repository

repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")

# Then can do:
repo.git_pull()
repo.git_add()
repo.git_commit()
repo.git_push()
repo.git_tag()
```
- Manages a local repository in a git-like manner
- Repository information here: [Repository docs link](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub#advanced-programmatic-repository-management)
- This seems a little weird. I'd rather just use the command line.

Git-based approach:
```
git clone https://huggingface.co/<namespace>/<your-model-id>
```
- Then just push to that repo and pull to it like any other git repo


## Building a Model Card
- Model card = the definition of a mode
    - Documents training and evaluation process
    - Helps others understand what to expect
    - Limitations, biases, and contexts in which model is and is not useful can be identified and understood
- Model card created in the `README.md` file and should include all the following:
    - Model description
    - Intended uses & limitations
    - How to use
    - Limitations and bias
    - Training data
    - Training procedure
    - Evaluation results

### Model description
- Basic details about the model
- Architecture, version, origional implementaiton, author, copyright
- Training procedures, parameters, and important disclaimers

### Intended uses & limitations
- What model is intended for, languages, fields, and domains where it cna be applied
- Can also document what's out of scope for the model and perform suboptimally

### How to use
- Examples on how to use the model
- Usage of model and tokenizer classes

### Training data
- Indicate which datasets model was trained on
- Brief description of the datasets too

### Training procedure
- All relevant aspects of training that are useful from a reproducibility perspective
- Any pre and post proccesing done on data
- Number of epochs, batch size, learning rate, etc.

### Variables and metrics
- What metrics used for evaluation and different factors you're measuring
- Mention which metrics are used, on which dataset and which dataset split 

### Evaluation results
- How well model performs on the evaluation dataset
- If model has a decision threshold either provide the decision threshold used in the evaluation at different thresholds for the intended uses

### Overall:
- Not required when publishing models but it will only benefit future users

