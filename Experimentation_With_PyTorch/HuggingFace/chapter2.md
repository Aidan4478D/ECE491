## Hugging Face Chapter 2 Notes

### Module 2 - Using Hugging Face Transformers

- [Behind the Pipeline](#behind-the-pipeline)

### Behind the Pipeline
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

```
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

