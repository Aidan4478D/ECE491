## Hugging Face Chapter 4 Notes

## Module 4 - Sharing Models & Tokenizers

- [The Hugging Face Hub](#the-hugging-face-hub)
- [Using Pretrained Models](#using-pretrained-models)

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
