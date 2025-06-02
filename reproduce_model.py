from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import Dataset
import numpy as np
import random


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

LEARNING_RATE = 5e-6
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

MODEL_NAME = "google/gemma-2-9b-it"
PROMPT = "Correct the following text, making only minimal changes where necessary."
max_seq_length = 512
output_dir = "./g9_gec"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation='eager')


# Change paths to proper directories
with open("FCE/fce_train_source.txt", "r") as file:
    source_fce = file.readlines()
    source_fce = [x.strip() for x in source_fce]

with open("FCE/fce_train_target.txt", "r") as file:
    target_fce = file.readlines()
    target_fce = [x.strip() for x in target_fce]

with open("BEA/bea_train_source.txt", "r") as file:
    source_bea = file.readlines()
    source_bea = [x.strip() for x in source_bea]

with open("BEA/bea_train_target.txt", "r") as file:
    target_bea = file.readlines()
    target_bea = [x.strip() for x in target_bea]

ds = Dataset.from_dict({
    "instruction": [PROMPT for x in range(len(source_fce))],
    "input": source_fce,
    "output": target_fce
})

correction_prompt = """{}

### Text to correct:
{}

### Corrected text:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = correction_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = ds.map(formatting_prompts_func, batched = True)

response_template = "### Corrected text:\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


# Single epoch on FCE
for x in range(1):
    model.train()
    current_source = []
    current_target = []

    for idx in range(len(source_fce)):
        if source_fce[idx] != target_fce[idx]:
            current_source.append(source_fce[idx])
            current_target.append(target_fce[idx])

            current_source.append(target_fce[idx])
            current_target.append(target_fce[idx])
        else:
            current_source.append(source_fce[idx])
            current_target.append(target_fce[idx])

    ds = Dataset.from_dict({
    "instruction": [PROMPT for x in range(len(current_source))],
    "input": current_source,
    "output": current_target
    })
    dataset = ds.map(formatting_prompts_func, batched = True)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        data_collator = collator,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            seed = 3407+x,
            save_strategy="no",
            output_dir = output_dir,
            per_device_train_batch_size = BATCH_SIZE,
            bf16 = True,
            learning_rate = LEARNING_RATE,
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            optim = "adamw_8bit",
            warmup_steps = 100,
            num_train_epochs=1,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            logging_steps=50
        ),
    )
    trainer_stats = trainer.train() 

ds = Dataset.from_dict({
    "instruction": [PROMPT for x in range(len(source_bea))],
    "input": source_bea,
    "output": target_bea
})
dataset = ds.map(formatting_prompts_func, batched = True)

# first epoch only errorneous examples, second epoch only examples without error with lower learning rate
for x in range(2):
    model.train()
    current_source = []
    current_target = []

    for idx in range(len(source_bea)):
        if source_bea[idx] != target_bea[idx]:
            current_source.append(source_bea[idx])
            current_target.append(target_bea[idx])

    if x == 1:
        # second epoch
        LEARNING_RATE = 0.0000003
        current_source = []
        current_target = []
        for idx in range(len(source_bea)):
            if source_bea[idx] == target_bea[idx]:
                current_source.append(source_bea[idx])
                current_target.append(target_bea[idx])

    ds = Dataset.from_dict({
    "instruction": [PROMPT for x in range(len(current_target))],
    "input": current_source,
    "output": current_target
    })
    dataset = ds.map(formatting_prompts_func, batched = True)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        data_collator = collator,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            seed = 3407+x,
            save_strategy="no",
            output_dir = output_dir,
            per_device_train_batch_size = BATCH_SIZE,
            bf16 = True,
            learning_rate = LEARNING_RATE,
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            optim = "adamw_8bit",
            num_train_epochs=1,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            logging_steps=50
        ),
    )
    trainer_stats = trainer.train()

    if x == 1:
        trainer.save_model(output_dir)
