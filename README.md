# Hierarchical Ensemble of Summarization Models
This repository contains the code for "[CUED at ProbSum 2023: Hierarchical Ensemble of Summarization Models](https://aclanthology.org/2023.bionlp-1.51/)" at the BioNLP Workshop @ ACL 2023.

## Token-level Ensemble
- This code combines the token-level probabilities (i.e., the output of the softmax layer) of multiple models.
- Currently, this only supports beamsearch decoding, as we modify `ensemble_beam_search` in the huggingface generation code. The code is in `ensemble_utils.py`

Example usage (see more details in [```examples_token_level_ensemble.ipynb```](examples_token_level_ensemble.ipynb)):

```python
# function to load a trained model
def load_model(path, device):
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

# this creates an ensemble of N models   
ensemble = [
    load_model(model1_path, device),
    load_model(model2_path, device),
    load_model(model3_path, device),
]

# inference call -- similar to model.generate() in the huggingface library
# all the arguments in model.generate() can be used there
# the input to each model can be different, so we define the input separately
# the input is input_ids which is created using the standard tokenization process
output_ids = ensemble_generate(
        ensemble,
        [inputs1.input_ids, inputs2.input_ids, inputs3.input_ids],
        num_beams=4,
        length_penalty=0.6,
        max_length=256,
        min_length=5,
        no_repeat_ngram_size=4,
)
output_txt = tokenizer.decode(output_ids(), skip_special_tokens=True)
```

## MBR Decoding
- MBR decoding essentially picks the output that maximises the expected reward.
- This combination is performed at the sequence level.

Example code in `mbr_decoding.py`

```
python mbr_decoding.py --filedir directory --outfile output.txt
```

In this example code, the directory should contain the output files (to be combined) where each file has one summary per line.


## Citation
```
@article{manakul2023cued,
  title={CUED at ProbSum 2023: Hierarchical Ensemble of Summarization Models},
  author={Manakul, Potsawee and Fathullah, Yassir and Liusie, Adian and Raina, Vyas and Raina, Vatsal and Gales, Mark},
  journal={arXiv preprint arXiv:2306.05317},
  year={2023}
}
```
