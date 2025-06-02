from transformers import AutoTokenizer
import torch
from get_fbeta_score import get_fbeta_score
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import random

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


MODEL_NAME = "g9_gec/"
IS_TOKENIZED = False

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
seed = 42
set_deterministic(seed=seed)


with open("BEA/bea_dev_source.txt", "r") as file:
    source_eval = file.readlines()
    source_eval = [x.strip() for x in source_eval]

with open("BEA/bea_test_source.txt", "r") as file:
    source_test = file.readlines()
    source_test = [x.strip() for x in source_test]

with open("CoNLL/conll_test_source.txt", "r") as file:
    conll_test = file.readlines()
    conll_test = [x.strip() for x in conll_test]

with open("JFLEG/jfleg_test_source.txt", "r") as file:
    jfleg_test = file.readlines()
    jfleg_test = [x.strip() for x in jfleg_test]

with open("JFLEG/jfleg_dev_source.txt", "r") as file:
    jfleg_dev = file.readlines()
    jfleg_dev = [x.strip() for x in jfleg_dev]


PROMPT = "Correct the following text, making only minimal changes where necessary."
correction_prompt = """{}

### Text to correct:
{}

### Corrected text:
{}"""


model = LLM(model=MODEL_NAME, tokenizer=MODEL_NAME, seed=1337, gpu_memory_utilization=0.95, tensor_parallel_size=1, enable_prefix_caching=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sampling_params = SamplingParams(n=1, temperature=0, top_k=1, max_tokens=1000, stop=["\n"])


input_datasets = [source_eval, source_test, conll_test, jfleg_test, jfleg_dev]
output_filenames = ["bea_dev.txt", "bea_test.txt", "conll_test.txt", "jfleg_test.txt", "jfleg_dev.txt"]

for input_dataset, output_filename in zip(input_datasets, output_filenames):
    input_tokens = []
    input_texts = [correction_prompt.format(
            PROMPT, # instruction
            t, # input
            "", # output - leave this blank for generation!
            ) for t in input_dataset]
    
    output = model.generate(input_texts, sampling_params)

    txt_output = []
    for o in output:
        txt_output.append(o.outputs[0].text)

    with open(f"Preds/{output_filename}", "w", encoding="utf-8") as file:
        file.writelines([x+"\n" for x in txt_output])
        